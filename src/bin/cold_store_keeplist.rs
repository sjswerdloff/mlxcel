//! Keep-list report for the v4 block cold store. **READ-ONLY. NOT DELETION
//! AUTHORITY.**
//!
//! Stuart, 2026-07-31: *"I need a way to know which data I should manually
//! delete... what I might end up having to do is list the X-Session-Ids that I
//! do not want deleted and get back the rest."*
//!
//! Revision 3, against Alden's re-review of `f4272bd`. See
//! `DESIGN_keeplist_report_20260802.md` — its normative body is revision-3
//! semantics; the superseded contract is quarantined in a historical section.
//!
//! # What this report is, precisely
//!
//! It answers one question: **which manifests were referenced only by sessions
//! the operator did not name?** That is an observation about *this invocation's
//! keep list*, not a claim that anything is safe to delete. An unnamed session
//! may be active, resumable, unknown to the operator, or simply forgotten.
//! Hence `UNPROTECTED`, and no byte figure is ever called "reclaimable".
//!
//! # Why a keep-list and not a delete-list
//!
//! The raw session key is never stored — the index filename is the key's digest
//! and the file carries only that digest. The store cannot name a conversation,
//! so *"show me what is deletable"* would offer an operator hex strings to
//! choose between. A keep-list requires identification **only of what is
//! kept**, which is the set the operator has.
//!
//! # The three buckets
//!
//! Per **manifest**, never per session — manifests are content-addressed and
//! prefixes are shared, so one manifest can be referenced by a kept session and
//! an unkept one at once.
//!
//! * **KEEP** — referenced by at least one named session.
//! * **UNPROTECTED** — referenced only by sessions not named in this run.
//! * **UNATTRIBUTED** — referenced by *no* session index. Never proposed, and
//!   its blocks join the protected side. Absence of an association is
//!   indistinguishable from a lost one.
//!
//! # Quiescence, and the limit of what this can prove
//!
//! Session index writes are serialised by a **process-local** `Mutex` in the
//! serving process (`record_session_manifest`), not by the cross-process
//! `flock` on `store.lock`. **A separate binary cannot exclude them and cannot
//! take a coherent snapshot.** So quiescence is asserted by the operator, never
//! proved here, and even a correct artifact is stale after any store mutation:
//! a later deletion boundary must revalidate the exact roots under authoritative
//! cross-process exclusion.

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mlxcel_core::cache::block_cold_store::BlockColdStore;
use sha2::{Digest, Sha256};

/// The v4 root directory name. `--store` is its PARENT: `BlockColdStore`
/// appends this itself, so every path this tool derives must go through
/// [`v4_root`] rather than joining onto `--store` directly. Revision 2 fixed
/// the validation and left two later joins at the wrong level.
const V4_ROOT_NAME: &str = "cold-storage-v4";

fn v4_root(store: &Path) -> PathBuf {
    store.join(V4_ROOT_NAME)
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// A short **pseudonymous** label for an operator-supplied session id.
///
/// NOT non-reversible, and revision 2 claimed it was. It is 48 bits of unsalted
/// SHA-256 over a possibly-guessable id, so it is an offline confirmation
/// oracle and it is stable across runs. It appears on the console only, to let
/// an operator tell two keep-list entries apart; it is **excluded from the
/// artifact**, which persists. (Alden, 2026-08-02.)
fn pseudonym(session_key: &str) -> String {
    hex(&BlockColdStore::key_digest_of(session_key))[..12].to_string()
}

/// Recursive byte total for one block directory.
///
/// `None` on any unreadable part, on any symlink or non-regular entry, and on
/// overflow. A symlink can point outside the store; a wrapped or saturated
/// total is a plausible number that reads as fact.
fn dir_bytes(path: &Path) -> Option<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![path.to_path_buf()];
    while let Some(p) = stack.pop() {
        for entry in std::fs::read_dir(&p).ok()? {
            let entry = entry.ok()?;
            let ft = entry.file_type().ok()?; // does not follow symlinks
            if ft.is_symlink() {
                return None;
            }
            if ft.is_dir() {
                stack.push(entry.path());
            } else if ft.is_file() {
                total = total.checked_add(entry.metadata().ok()?.len())?;
            } else {
                return None;
            }
        }
    }
    Some(total)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// SHA-256 of the running binary.
///
/// Revision 2 recorded `option_env!("MLXCEL_GIT_SHA").unwrap_or("unknown")`,
/// which ordinary builds do not set — so every artifact would have claimed
/// provenance it did not have. Hashing the executable pins the exact code that
/// produced the artifact, including uncommitted changes, which a commit id
/// cannot. Failure to compute it refuses the artifact rather than degrading to
/// a placeholder.
fn running_binary_digest() -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let bytes = std::fs::read(&exe).map_err(|e| format!("read {}: {e}", exe.display()))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(hex(&h.finalize()))
}

pub struct Args {
    pub store: PathBuf,
    pub keep: Vec<String>,
    pub keep_none: bool,
    pub artifact: Option<PathBuf>,
    /// Set only by tests. Production always probes.
    pub skip_server_probe: bool,
}

fn usage() -> String {
    "\
cold-store-keeplist — READ-ONLY report. Deletes nothing. NOT deletion authority.

USAGE:
  cold-store-keeplist --store <DIR-CONTAINING-cold-storage-v4> \\
      --store-is-quiescent --keep-file <FILE> [--artifact <OUT>]

  --store <DIR>           the directory CONTAINING 'cold-storage-v4'
  --keep-file <FILE>      session ids to PROTECT, one per line; '#' comments.
                          Preferred over --keep: argv is world-readable.
  --keep <SESSION-ID>     one id to protect; repeatable.
  --keep-none             assert a deliberately EMPTY keep list. Refused if any
                          id is also supplied — the two assertions contradict.
  --store-is-quiescent    REQUIRED. Asserts no server is writing this store.
                          This tool CANNOT verify it.
  --artifact <OUT>        exact manifest and block hashes with provenance, as
                          evidence for a LATER, separate deletion decision.
                          Refuses to clobber, refuses a path inside the store.

EXIT CODES:
  0  report complete; every named session matched an index
  1  report complete, but a named session matched nothing (no artifact written)
  3  report INCOMPLETE — figures withheld; do not act on this run
  2  usage, validation, or I/O error

EVIDENCE, NOT A GRANT. It says what nothing in THIS keep list protects. Even a
correct artifact is stale after any store mutation.
"
    .to_string()
}

pub fn parse_args(argv: Vec<String>) -> Result<Args, String> {
    let mut store: Option<PathBuf> = None;
    let mut keep: Vec<String> = Vec::new();
    let (mut keep_none, mut quiescent) = (false, false);
    let mut artifact: Option<PathBuf> = None;
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--store" => store = Some(PathBuf::from(it.next().ok_or("--store needs a directory")?)),
            "--keep" => keep.push(it.next().ok_or("--keep needs a session id")?),
            "--keep-file" => {
                let f = it.next().ok_or("--keep-file needs a path")?;
                let body = std::fs::read_to_string(&f)
                    .map_err(|e| format!("cannot read keep-file {f}: {e}"))?;
                for line in body.lines() {
                    let line = line.trim();
                    if !line.is_empty() && !line.starts_with('#') {
                        keep.push(line.to_string());
                    }
                }
            }
            "--keep-none" => keep_none = true,
            "--store-is-quiescent" => quiescent = true,
            "--artifact" => {
                artifact = Some(PathBuf::from(it.next().ok_or("--artifact needs a path")?))
            }
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown argument {other}\n\n{}", usage())),
        }
    }
    let store = store.ok_or_else(|| format!("--store is required\n\n{}", usage()))?;

    let v4 = v4_root(&store);
    if !v4.is_dir() {
        return Err(format!(
            "--store must be the directory CONTAINING '{V4_ROOT_NAME}', not the v4 root \
             itself.\nLooked for: {}\n\n\
             The store appends '{V4_ROOT_NAME}' internally. A wrong level enumerates \
             nothing and prints as a clean zero report.",
            v4.display()
        ));
    }
    for sub in ["manifests", "sessions", "blocks"] {
        if !v4.join(sub).is_dir() {
            return Err(format!(
                "{} is not a directory — is this a v4 store?",
                v4.join(sub).display()
            ));
        }
    }
    if !quiescent {
        return Err("--store-is-quiescent is required.\n\n\
             This tool CANNOT verify quiescence: session index writes are guarded by a \
             process-local mutex in the serving process, not by the cross-process store \
             lock, so a separate binary cannot exclude them. A scan taken while the \
             server writes can classify a manifest a NAMED conversation depends on as \
             unprotected.\n\nStop the server, then re-run with --store-is-quiescent."
            .to_string());
    }
    if keep_none && !keep.is_empty() {
        return Err(format!(
            "--keep-none was passed alongside {} session id(s). Those assertions \
             contradict: one says protect nothing deliberately, the other names things \
             to protect.",
            keep.len()
        ));
    }
    if keep.is_empty() && !keep_none {
        return Err("empty keep list refused.\n\n\
             With no protected sessions, every attributed manifest is reported as \
             unprotected. An unset shell variable or an empty keep-file produces that \
             silently. If an empty list is what you mean, pass --keep-none."
            .to_string());
    }
    let mut seen = HashSet::new();
    let dupes = keep.iter().filter(|k| !seen.insert((*k).clone())).count();
    if dupes > 0 {
        return Err(format!(
            "{dupes} duplicate session id(s). Duplicates inflate the named count while \
             the matched count deduplicates, so the two disagree for a reason that is \
             not a finding."
        ));
    }
    Ok(Args {
        store,
        keep,
        keep_none,
        artifact,
        skip_server_probe: false,
    })
}

/// Result of the cheap contradiction check on the quiescence assertion.
///
/// **Fails closed.** `pgrep` exits 1 for "no match", a real negative; any other
/// outcome means the check could not run, and a check that could not run must
/// not be reported as one that found nothing.
#[derive(Debug, PartialEq, Eq)]
pub enum Quiescence {
    NoServerFound,
    ServerRunning,
    CouldNotCheck(String),
}

pub fn classify_probe_output(spawn: Result<std::process::Output, std::io::Error>) -> Quiescence {
    match spawn {
        Err(e) => Quiescence::CouldNotCheck(format!("could not run pgrep: {e}")),
        Ok(o) => match o.status.code() {
            Some(0) if !o.stdout.is_empty() => Quiescence::ServerRunning,
            Some(1) => Quiescence::NoServerFound,
            other => Quiescence::CouldNotCheck(format!(
                "pgrep exited {other:?}, which is neither a match nor a clean no-match"
            )),
        },
    }
}

fn probe_for_live_server() -> Quiescence {
    classify_probe_output(
        std::process::Command::new("/usr/bin/pgrep")
            .arg("-x")
            .arg("mlxcel-server")
            .output(),
    )
}

pub struct Buckets {
    pub keep: Vec<[u8; 32]>,
    pub unprotected: Vec<[u8; 32]>,
    pub unattributed: Vec<[u8; 32]>,
    pub matched_digests: HashSet<[u8; 32]>,
}

/// The classification, as a pure function over already-read state.
pub fn classify(
    indexes: &[([u8; 32], Vec<[u8; 32]>)],
    manifest_hashes: &[[u8; 32]],
    keep_digests: &HashSet<[u8; 32]>,
) -> Buckets {
    let mut by_kept: HashSet<[u8; 32]> = HashSet::new();
    let mut by_unkept: HashSet<[u8; 32]> = HashSet::new();
    let mut matched_digests: HashSet<[u8; 32]> = HashSet::new();

    for (digest, manifests) in indexes {
        let kept = keep_digests.contains(digest);
        if kept {
            matched_digests.insert(*digest);
        }
        for m in manifests {
            if kept {
                by_kept.insert(*m);
            } else {
                by_unkept.insert(*m);
            }
        }
    }

    let (mut keep, mut unprotected, mut unattributed) = (Vec::new(), Vec::new(), Vec::new());
    for h in manifest_hashes {
        // ORDER IS THE SAFETY PROPERTY. Kept wins, so a manifest shared between
        // a protected conversation and an unprotected one stays protected.
        // Shared prefixes are the common case — that is what content addressing
        // is for — so reversing these arms would report most of a store as
        // unprotected.
        if by_kept.contains(h) {
            keep.push(*h);
        } else if by_unkept.contains(h) {
            unprotected.push(*h);
        } else {
            unattributed.push(*h);
        }
    }
    keep.sort_unstable();
    unprotected.sort_unstable();
    unattributed.sort_unstable();
    Buckets {
        keep,
        unprotected,
        unattributed,
        matched_digests,
    }
}

/// Everything the report determined. Returned rather than printed so the whole
/// pipeline is reachable by a test — revision 2's guards were only ever
/// exercised by hand.
pub struct Report {
    pub buckets: Buckets,
    pub candidate_blocks: Vec<[u8; 32]>,
    pub bytes: Option<u64>,
    pub unsized_blocks: usize,
    pub sessions_enumerated: usize,
    /// Indices into `Args::keep` that matched nothing.
    pub unmatched: Vec<usize>,
}

pub enum Outcome {
    Complete(Report),
    /// Report produced, but a named session matched nothing. No artifact.
    Unmatched(Report),
    /// Figures withheld; nothing here may be acted on.
    Incomplete(String),
    Refused(String),
}

pub fn run(args: &Args) -> Outcome {
    if !args.skip_server_probe {
        match probe_for_live_server() {
            Quiescence::ServerRunning => {
                return Outcome::Refused(
                    "an mlxcel-server process is running. --store-is-quiescent was \
                     asserted and is contradicted. Stop the server and re-run.\n\
                     (The reverse does not hold: no detected server is NOT proof.)"
                        .into(),
                )
            }
            Quiescence::CouldNotCheck(why) => {
                return Outcome::Refused(format!(
                    "the contradiction check on --store-is-quiescent could not run \
                     ({why}). A check that did not run is not a check that found nothing."
                ))
            }
            Quiescence::NoServerFound => {}
        }
    }

    // The runtime fingerprint participates in manifest ADDRESSING, not in
    // reading one by hash. A report that enumerates and reads by hash never
    // consults it.
    let store = BlockColdStore::new(args.store.clone(), [0u8; 32]);

    let raw = match store.enumerate_session_indexes() {
        Ok(v) => v,
        Err(e) => return Outcome::Refused(format!("cannot enumerate session indexes: {e}")),
    };

    // PATH/HEADER AGREEMENT, on the path actually enumerated.
    //
    // Revision 2 derived the filename that SHOULD hold the header digest and
    // checked whether such a file existed — a different question, which `a.idx`
    // claiming digest `b` passes whenever `b.idx` also exists. The stem of the
    // file we read is the only independent witness to its identity.
    let mut indexes: Vec<([u8; 32], Vec<[u8; 32]>)> = Vec::new();
    for entry in raw {
        let Some(digest) = entry.key_digest else {
            return Outcome::Incomplete(
                "a session index vanished between the directory scan and the read. \
                 The store is not quiescent, or the scan raced a writer."
                    .into(),
            );
        };
        let stem = entry
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if stem.len() != 64 || !stem.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Outcome::Refused(format!(
                "{} does not have a 64-hex filename stem, so its session identity \
                 cannot be established.",
                entry.path.display()
            ));
        }
        if stem != hex(&digest) {
            return Outcome::Refused(format!(
                "{} claims digest {} but its filename says {}. The file's session \
                 identity is contradicted, and associations authorize deletion.",
                entry.path.display(),
                hex(&digest),
                stem
            ));
        }
        indexes.push((
            digest,
            entry.associations.iter().map(|a| a.manifest_hash).collect(),
        ));
    }

    let manifest_hashes = match store.enumerate_manifest_hashes() {
        Ok(v) => v,
        Err(e) => return Outcome::Refused(format!("cannot enumerate manifests: {e}")),
    };

    let keep_digests: HashSet<[u8; 32]> = args
        .keep
        .iter()
        .map(|k| BlockColdStore::key_digest_of(k))
        .collect();
    let buckets = classify(&indexes, &manifest_hashes, &keep_digests);

    // Block reachability against a HYPOTHETICAL root set. Nothing is marked,
    // tombstoned or written. UNATTRIBUTED joins the PROTECTED side: a manifest
    // whose association was merely lost may belong to a live conversation.
    let mut protected: HashSet<[u8; 32]> = HashSet::new();
    let mut unprotected_blocks: HashSet<[u8; 32]> = HashSet::new();
    let mut unreadable = 0usize;
    let mut gather = |set: &[[u8; 32]], into: &mut HashSet<[u8; 32]>| {
        for h in set {
            match store.read_manifest(h) {
                Ok(m) => into.extend(m.block_hashes.iter().copied()),
                Err(_) => unreadable += 1,
            }
        }
    };
    gather(&buckets.keep, &mut protected);
    gather(&buckets.unattributed, &mut protected);
    gather(&buckets.unprotected, &mut unprotected_blocks);
    drop(gather);
    if unreadable > 0 {
        return Outcome::Incomplete(format!(
            "{unreadable} manifest(s) unreadable, so reachability cannot be \
             established. No block or byte figure is produced."
        ));
    }

    let mut candidate_blocks: Vec<[u8; 32]> =
        unprotected_blocks.difference(&protected).copied().collect();
    candidate_blocks.sort_unstable();

    let blocks_dir = v4_root(&args.store).join("blocks");
    let (mut total, mut unsized_blocks) = (Some(0u64), 0usize);
    for blk in &candidate_blocks {
        match dir_bytes(&blocks_dir.join(hex(blk))) {
            // checked, not saturating: a saturated total is a plausible maximum
            // that reads as a measurement.
            Some(n) => total = total.and_then(|t| t.checked_add(n)),
            None => unsized_blocks += 1,
        }
    }
    let bytes = if unsized_blocks > 0 { None } else { total };

    let unmatched: Vec<usize> = args
        .keep
        .iter()
        .enumerate()
        .filter(|(_, k)| !buckets.matched_digests.contains(&BlockColdStore::key_digest_of(k)))
        .map(|(i, _)| i)
        .collect();

    let report = Report {
        buckets,
        candidate_blocks,
        bytes,
        unsized_blocks,
        sessions_enumerated: indexes.len(),
        unmatched,
    };
    if !report.unmatched.is_empty() {
        return Outcome::Unmatched(report);
    }
    Outcome::Complete(report)
}

/// Write the evidence artifact.
///
/// Called ONLY on a complete report with no unmatched keep entries. Revision 2
/// wrote it before the unmatched check, so a typo produced a detached file
/// saying `completeness=complete` while protecting nothing.
pub fn write_artifact(path: &Path, args: &Args, r: &Report) -> Result<(), String> {
    let canon_store = v4_root(&args.store)
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize store: {e}"))?;
    if let Some(parent) = path.parent() {
        if let Ok(p) = parent.canonicalize() {
            if p.starts_with(&canon_store) {
                return Err(
                    "--artifact points inside the store. Evidence about a store must \
                     not be written into the thing it describes."
                        .into(),
                );
            }
        }
    }
    if path.symlink_metadata().is_ok() {
        return Err(format!(
            "{} already exists. Refusing to clobber: a later run would silently \
             replace the exact object a human reviewed.",
            path.display()
        ));
    }
    let code = running_binary_digest()?;

    let mut out = String::new();
    out.push_str("# cold-store-keeplist artifact — EVIDENCE, NOT A DELETION GRANT\n");
    out.push_str("# Stale after ANY store mutation. A deletion boundary must revalidate\n");
    out.push_str("# these exact roots under authoritative cross-process exclusion.\n");
    out.push_str(&format!("store\t{}\n", canon_store.display()));
    out.push_str(&format!("binary_sha256\t{code}\n"));
    out.push_str("quiescence\toperator-asserted\n");
    out.push_str(&format!("sessions_enumerated\t{}\n", r.sessions_enumerated));
    out.push_str(&format!("keep_named\t{}\n", args.keep.len()));
    out.push_str(&format!("keep_matched\t{}\n", r.buckets.matched_digests.len()));
    out.push_str("keep_unmatched\t0\n");
    out.push_str("completeness\tcomplete\n");
    // Session pseudonyms are deliberately absent: a stable 48-bit digest prefix
    // of a guessable id is an offline confirmation oracle, and this file persists.
    for h in &r.buckets.keep {
        out.push_str(&format!("keep_manifest\t{}\n", hex(h)));
    }
    for h in &r.buckets.unattributed {
        out.push_str(&format!("unattributed_manifest\t{}\n", hex(h)));
    }
    for h in &r.buckets.unprotected {
        out.push_str(&format!("unprotected_manifest\t{}\n", hex(h)));
    }
    for b in &r.candidate_blocks {
        out.push_str(&format!("candidate_block\t{}\n", hex(b)));
    }
    if let Some(n) = r.bytes {
        out.push_str(&format!("candidate_bytes\t{n}\n"));
    }
    let mut h = Sha256::new();
    h.update(out.as_bytes());
    out.push_str(&format!("artifact_sha256\t{}\n", hex(&h.finalize())));

    // Create-new + 0600, fsync, then rename onto a path verified absent above.
    let tmp = path.with_extension("partial");
    let _ = std::fs::remove_file(&tmp);
    let mut f = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&tmp)
        .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
    f.write_all(out.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    f.sync_all().map_err(|e| format!("fsync: {e}"))?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| format!("publish: {e}"))
}

use std::os::unix::fs::OpenOptionsExt as _;

fn print_report(args: &Args, r: &Report) {
    println!("cold store    {}", v4_root(&args.store).display());
    println!("quiescence    OPERATOR-ASSERTED, not verified by this tool");
    println!(
        "sessions      {} indexes; {} named{}, {} matched",
        r.sessions_enumerated,
        args.keep.len(),
        if args.keep_none { " (--keep-none)" } else { "" },
        r.buckets.matched_digests.len()
    );
    println!();
    println!("KEEP          {:>6} manifests", r.buckets.keep.len());
    println!(
        "UNPROTECTED   {:>6} manifests   → {} blocks referenced by no KEEP or \
         UNATTRIBUTED manifest",
        r.buckets.unprotected.len(),
        r.candidate_blocks.len()
    );
    println!(
        "UNATTRIBUTED  {:>6} manifests   [protected; never proposed]",
        r.buckets.unattributed.len()
    );
    println!();
    match r.bytes {
        Some(n) => println!("those blocks occupy {:.2} GiB on disk", gib(n)),
        None => println!(
            "SIZE WITHHELD: {} block(s) could not be sized, or the total overflowed",
            r.unsized_blocks
        ),
    }
    println!();
    println!(
        "UNPROTECTED means: nothing in THIS keep list references it. It does NOT mean \
         safe to delete.\nAn unnamed session may be active, resumable, or simply \
         forgotten."
    );
}

fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1).collect()) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Outcome::Refused(why) => {
            eprintln!("REFUSING: {why}");
            ExitCode::from(2)
        }
        Outcome::Incomplete(why) => {
            eprintln!("REPORT INCOMPLETE: {why} Do not act on this run.");
            ExitCode::from(3)
        }
        Outcome::Unmatched(r) => {
            print_report(&args, &r);
            println!();
            println!("UNMATCHED — these protected nothing (pseudonyms, not anonymous):");
            for i in &r.unmatched {
                println!("  keep-list entry {}  ({})", i + 1, pseudonym(&args.keep[*i]));
            }
            println!("\nNO ARTIFACT WRITTEN: a keep list with an entry that protects");
            println!("nothing is not evidence anyone should act on.");
            ExitCode::from(1)
        }
        Outcome::Complete(r) => {
            print_report(&args, &r);
            if let Some(path) = &args.artifact {
                match write_artifact(path, &args, &r) {
                    Ok(()) => println!("\nartifact      {}", path.display()),
                    Err(e) => {
                        eprintln!("cannot write artifact: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            if r.unsized_blocks > 0 {
                return ExitCode::from(3);
            }
            ExitCode::SUCCESS
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel_core::cache::block_cold_store::Manifest;

    fn h(n: u8) -> [u8; 32] {
        [n; 32]
    }

    // ---------- pure classifier ----------

    fn idx(d: u8, manifests: &[u8]) -> ([u8; 32], Vec<[u8; 32]>) {
        (h(d), manifests.iter().map(|m| h(*m)).collect())
    }

    /// THE SAFETY PROPERTY. MUTATION: swap the two arms in `classify`.
    #[test]
    fn shared_manifest_is_protected_by_any_keeper() {
        let keep: HashSet<[u8; 32]> = [h(10)].into_iter().collect();
        let b = classify(
            &[idx(10, &[1, 2]), idx(20, &[1, 3])],
            &[h(1), h(2), h(3)],
            &keep,
        );
        assert!(b.keep.contains(&h(1)), "shared manifest must be KEPT");
        assert_eq!(b.unprotected, vec![h(3)]);
    }

    /// MUTATION: change the final `else` arm to push into `unprotected`.
    #[test]
    fn unreferenced_manifest_is_unattributed() {
        let b = classify(&[idx(20, &[1])], &[h(1), h(99)], &HashSet::new());
        assert_eq!(b.unprotected, vec![h(1)]);
        assert_eq!(b.unattributed, vec![h(99)]);
    }

    #[test]
    fn buckets_partition_the_manifest_set_exactly() {
        let all = [h(1), h(2), h(3), h(4), h(5)];
        let keep: HashSet<[u8; 32]> = [h(10)].into_iter().collect();
        let b = classify(&[idx(10, &[1]), idx(20, &[2, 3])], &all, &keep);
        assert_eq!(
            b.keep.len() + b.unprotected.len() + b.unattributed.len(),
            all.len()
        );
        let u: HashSet<_> = b
            .keep
            .iter()
            .chain(&b.unprotected)
            .chain(&b.unattributed)
            .copied()
            .collect();
        assert_eq!(u.len(), all.len());
    }

    #[test]
    fn bucket_output_is_sorted_for_artifact_determinism() {
        let b = classify(&[idx(20, &[5, 1, 3])], &[h(5), h(1), h(3)], &HashSet::new());
        assert_eq!(b.unprotected, vec![h(1), h(3), h(5)]);
    }

    // ---------- quiescence probe, all three states ----------

    fn out(code: i32, stdout: &str) -> std::process::Output {
        use std::os::unix::process::ExitStatusExt;
        std::process::Output {
            status: std::process::ExitStatus::from_raw(code << 8),
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
        }
    }

    #[test]
    fn probe_match_means_server_running() {
        assert_eq!(
            classify_probe_output(Ok(out(0, "23220\n"))),
            Quiescence::ServerRunning
        );
    }

    #[test]
    fn probe_exit_one_is_a_clean_no_match() {
        assert_eq!(classify_probe_output(Ok(out(1, ""))), Quiescence::NoServerFound);
    }

    /// The failure that must NOT look like a pass. MUTATION: collapse the
    /// `other` arm into `NoServerFound` and this goes red.
    #[test]
    fn probe_error_is_could_not_check_not_no_server() {
        assert!(matches!(
            classify_probe_output(Ok(out(2, ""))),
            Quiescence::CouldNotCheck(_)
        ));
        assert!(matches!(
            classify_probe_output(Err(std::io::Error::other("boom"))),
            Quiescence::CouldNotCheck(_)
        ));
    }

    // ---------- real on-disk fixtures ----------

    /// Build a real v4 store using only the store's own public writers, so the
    /// fixture cannot drift from the production format.
    struct Fixture {
        _dir: tempfile::TempDir,
        base: PathBuf,
    }

    fn manifest(blocks: &[u8], tag: &str) -> Manifest {
        Manifest {
            runtime_fingerprint: [7u8; 32],
            model_id: format!("fixture-{tag}"),
            template_sig: "sig".into(),
            block_size: 16,
            block_hashes: blocks.iter().map(|b| h(*b)).collect(),
            prompt_len: 1,
            total_tokens: 1,
            timestamp_nanos: 1,
        }
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_path_buf();
        for sub in ["manifests", "sessions", "blocks"] {
            std::fs::create_dir_all(v4_root(&base).join(sub)).unwrap();
        }
        Fixture { _dir: dir, base }
    }

    /// Install block directories so `write_manifest` will publish.
    ///
    /// `write_manifest` refuses a manifest whose blocks are absent at
    /// publication time — a real invariant, and the fixture must satisfy it
    /// rather than route around it.
    ///
    /// LIMIT, stated because it bounds what these tests prove: the directories
    /// carry no payload. That is sufficient here because this report only
    /// enumerates manifests and SIZES block directories — it never reads block
    /// contents — and it is insufficient for any test of block I/O.
    fn install_blocks(base: &Path, blocks: &[u8], bytes_each: usize) {
        for b in blocks {
            let d = v4_root(base).join("blocks").join(hex(&h(*b)));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("payload"), vec![0u8; bytes_each]).unwrap();
        }
    }

    fn args_for(base: &Path, keep: &[&str]) -> Args {
        Args {
            store: base.to_path_buf(),
            keep: keep.iter().map(|s| s.to_string()).collect(),
            keep_none: keep.is_empty(),
            artifact: None,
            skip_server_probe: true,
        }
    }

    /// The happy path, end to end, against a real store written by the store's
    /// own API. Revision 2 had never run this — every index on the only real
    /// store was pre-seal, so a complete report had never been produced.
    #[test]
    fn happy_path_over_a_real_store() {
        let f = fixture();
        install_blocks(&f.base, &[1, 2, 3, 4], 1024);
        let store = BlockColdStore::new(f.base.clone(), [7u8; 32]);
        let kept = manifest(&[1, 2], "kept");
        let other = manifest(&[3], "other");
        let orphan = manifest(&[4], "orphan");
        store.write_manifest(&kept).unwrap();
        store.write_manifest(&other).unwrap();
        store.write_manifest(&orphan).unwrap();
        store.record_session_manifest("keep-me", &kept.hash()).unwrap();
        store.record_session_manifest("drop-me", &other.hash()).unwrap();

        let a = args_for(&f.base, &["keep-me"]);
        let Outcome::Complete(r) = run(&a) else {
            panic!("expected a complete report");
        };
        assert_eq!(r.buckets.keep, vec![kept.hash()]);
        assert_eq!(r.buckets.unprotected, vec![other.hash()]);
        assert_eq!(r.buckets.unattributed, vec![orphan.hash()]);
        assert_eq!(r.sessions_enumerated, 2);
        assert!(r.unmatched.is_empty());
    }

    /// P0 from Alden's re-review. `a.idx` claiming `b`'s digest must be
    /// refused. Revision 2 only checked that SOME file named for the header
    /// digest existed, which this arrangement satisfies.
    ///
    /// MUTATION: drop the `stem != hex(&digest)` arm in `run` and this goes red.
    #[test]
    fn index_whose_filename_contradicts_its_header_is_refused() {
        let f = fixture();
        install_blocks(&f.base, &[1], 512);
        let store = BlockColdStore::new(f.base.clone(), [7u8; 32]);
        let m = manifest(&[1], "m");
        store.write_manifest(&m).unwrap();
        store.record_session_manifest("alpha", &m.hash()).unwrap();
        store.record_session_manifest("beta", &m.hash()).unwrap();

        // Copy alpha's file over a THIRD name. Both real files still exist, so
        // revision 2's "does a file of that name exist" check would pass.
        let sessions = v4_root(&f.base).join("sessions");
        let alpha = sessions.join(format!(
            "{}.idx",
            hex(&BlockColdStore::key_digest_of("alpha"))
        ));
        let impostor = sessions.join(format!("{}.idx", "c".repeat(64)));
        std::fs::copy(&alpha, &impostor).unwrap();

        match run(&args_for(&f.base, &["alpha"])) {
            Outcome::Refused(why) => assert!(
                why.contains("claims digest") && why.contains("filename says"),
                "unexpected refusal: {why}"
            ),
            _ => panic!("an index whose stem contradicts its header must be refused"),
        }
    }

    /// A symlinked index must be refused at enumeration, where the distinction
    /// is still visible.
    #[test]
    fn symlinked_index_is_refused() {
        let f = fixture();
        install_blocks(&f.base, &[1], 512);
        let store = BlockColdStore::new(f.base.clone(), [7u8; 32]);
        let m = manifest(&[1], "m");
        store.write_manifest(&m).unwrap();
        store.record_session_manifest("alpha", &m.hash()).unwrap();
        let sessions = v4_root(&f.base).join("sessions");
        let alpha = sessions.join(format!(
            "{}.idx",
            hex(&BlockColdStore::key_digest_of("alpha"))
        ));
        std::os::unix::fs::symlink(&alpha, sessions.join(format!("{}.idx", "d".repeat(64))))
            .unwrap();
        assert!(matches!(run(&args_for(&f.base, &["alpha"])), Outcome::Refused(_)));
    }

    /// A typo'd keep id yields Unmatched, and NO artifact may be produced.
    /// Revision 2 wrote a `completeness=complete` artifact before this check.
    #[test]
    fn unmatched_keep_entry_yields_no_artifact() {
        let f = fixture();
        install_blocks(&f.base, &[1], 512);
        let store = BlockColdStore::new(f.base.clone(), [7u8; 32]);
        let m = manifest(&[1], "m");
        store.write_manifest(&m).unwrap();
        store.record_session_manifest("real", &m.hash()).unwrap();

        let out_path = f.base.join("artifact.tsv");
        let mut a = args_for(&f.base, &["real", "typoo"]);
        a.artifact = Some(out_path.clone());
        let o = run(&a);
        assert!(matches!(o, Outcome::Unmatched(_)));
        // `main` only calls write_artifact on Complete; assert the file is absent.
        assert!(!out_path.exists(), "no artifact may exist for an unmatched run");
    }

    #[test]
    fn artifact_refuses_to_clobber_and_seals_itself() {
        let f = fixture();
        install_blocks(&f.base, &[1], 512);
        let store = BlockColdStore::new(f.base.clone(), [7u8; 32]);
        let m = manifest(&[1], "m");
        store.write_manifest(&m).unwrap();
        store.record_session_manifest("real", &m.hash()).unwrap();
        let a = args_for(&f.base, &["real"]);
        let Outcome::Complete(r) = run(&a) else {
            panic!("expected complete")
        };

        let dest = f.base.join("evidence.tsv");
        write_artifact(&dest, &a, &r).expect("first write");
        let body = std::fs::read_to_string(&dest).unwrap();
        assert!(body.contains("artifact_sha256\t"));
        assert!(body.contains("binary_sha256\t"));
        assert!(!body.contains("keep_label"), "pseudonyms must not persist");
        assert!(write_artifact(&dest, &a, &r).is_err(), "must not clobber");
    }

    #[test]
    fn artifact_inside_the_store_is_refused() {
        let f = fixture();
        install_blocks(&f.base, &[1], 512);
        let store = BlockColdStore::new(f.base.clone(), [7u8; 32]);
        let m = manifest(&[1], "m");
        store.write_manifest(&m).unwrap();
        store.record_session_manifest("real", &m.hash()).unwrap();
        let a = args_for(&f.base, &["real"]);
        let Outcome::Complete(r) = run(&a) else {
            panic!("expected complete")
        };
        let inside = v4_root(&f.base).join("sessions").join("evidence.tsv");
        assert!(write_artifact(&inside, &a, &r).is_err());
    }

    // ---------- argument contract ----------

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn argument_guards_refuse_the_dangerous_shapes() {
        let f = fixture();
        let s = f.base.to_str().unwrap();
        // quiescence not asserted
        assert!(parse_args(argv(&["--store", s, "--keep", "a"])).is_err());
        // empty keep without --keep-none
        assert!(parse_args(argv(&["--store", s, "--store-is-quiescent"])).is_err());
        // contradictory assertions
        assert!(parse_args(argv(&[
            "--store", s, "--store-is-quiescent", "--keep-none", "--keep", "a"
        ]))
        .is_err());
        // duplicates
        assert!(parse_args(argv(&[
            "--store", s, "--store-is-quiescent", "--keep", "a", "--keep", "a"
        ]))
        .is_err());
        // the v4 root itself, rather than its parent
        let v4 = v4_root(&f.base);
        assert!(parse_args(argv(&[
            "--store",
            v4.to_str().unwrap(),
            "--store-is-quiescent",
            "--keep-none"
        ]))
        .is_err());
        // the correct shape
        assert!(parse_args(argv(&["--store", s, "--store-is-quiescent", "--keep-none"])).is_ok());
    }

    #[test]
    fn key_digest_is_stable_and_distinguishing() {
        let a = BlockColdStore::key_digest_of("session-a");
        assert_eq!(a, BlockColdStore::key_digest_of("session-a"));
        assert_ne!(a, BlockColdStore::key_digest_of("session-b"));
    }
}
