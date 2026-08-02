//! Keep-list report for the v4 block cold store. **READ-ONLY. NOT DELETION
//! AUTHORITY.**
//!
//! Stuart, 2026-07-31: *"I need a way to know which data I should manually
//! delete... what I might end up having to do is list the X-Session-Ids that I
//! do not want deleted and get back the rest."*
//!
//! Revised against Alden's review of `84edcb4`, which returned four P0s. See
//! `DESIGN_keeplist_report_20260802.md`.
//!
//! # What this report is, precisely
//!
//! It answers one question: **which manifests were referenced only by sessions
//! the operator did not name?** That is an observation about *this invocation's
//! keep list*, not a claim that anything is safe to delete. An unnamed session
//! may be active, resumable, unknown to the operator, or simply forgotten.
//! Hence `UNPROTECTED` rather than `RELEASABLE`, and no byte figure is ever
//! labelled "reclaimable".
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
//! * **UNATTRIBUTED** — referenced by *no* session index. Never proposed for
//!   anything, and its blocks are grouped with the protected ones. Absence of
//!   an association is indistinguishable from a lost one: persisted before
//!   session tracking, lost to the crash window, or an index whose digest did
//!   not match.
//!
//! # Quiescence, and the limit of what this can prove
//!
//! Session index writes are serialised by a **process-local** `Mutex` in the
//! serving process (`record_session_manifest`), not by the cross-process
//! `flock` on `store.lock`. **A separate binary therefore cannot exclude them
//! and cannot take a coherent snapshot.** Alden's interleaving: this report
//! reads unkept session U referencing manifest M; kept session K then adds a
//! reference to M; this report enumerates M and classifies it from the stale
//! read — and M lands in UNPROTECTED while a named conversation depends on it.
//!
//! So quiescence is **asserted by the operator, not proved by this tool**, via
//! `--store-is-quiescent`, and the assertion is recorded in the artifact. A
//! cheap contradiction check refuses the run if a live `mlxcel-server` is
//! detected; a negative result there is not proof of anything.
//!
//! The real fix is to extend the cross-process lock to cover session index
//! writes. That is a change to the serving path and not this tool's to make.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mlxcel_core::cache::block_cold_store::BlockColdStore;

/// The v4 root directory name, used to validate that `--store` names the level
/// the store expects. A wrong level enumerates nothing and prints as a clean
/// zero report — the one output an operator reads as "nothing to do".
const V4_ROOT_NAME: &str = "cold-storage-v4";

fn hex32(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// A short, non-reversible label for an operator-supplied session id.
///
/// Unmatched ids were previously printed verbatim, contradicting this tool's
/// own claim that raw ids never reach the report body. The operator identifies
/// an entry by its input line; the label only confirms which digest was used.
fn short_label(session_key: &str) -> String {
    hex32(&BlockColdStore::key_digest_of(session_key))[..12].to_string()
}

/// Recursive byte total for one block directory.
///
/// `None` on any unreadable part, and on any symlink or non-regular entry: a
/// symlink can point outside the store, and following it would attribute
/// someone else's bytes to this report. Overflow is checked rather than
/// wrapping — a wrapped total is a small number that reads as good news.
fn dir_bytes(path: &Path) -> Option<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![path.to_path_buf()];
    while let Some(p) = stack.pop() {
        for entry in std::fs::read_dir(&p).ok()? {
            let entry = entry.ok()?;
            // `DirEntry::file_type` does not follow symlinks.
            let ft = entry.file_type().ok()?;
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

struct Args {
    store: PathBuf,
    keep: Vec<String>,
    keep_none: bool,
    artifact: Option<PathBuf>,
}

fn usage() -> String {
    "\
cold-store-keeplist — READ-ONLY report. Deletes nothing. NOT deletion authority.

USAGE:
  cold-store-keeplist --store <V4-ROOT> --store-is-quiescent \\
      --keep-file <FILE> [--artifact <OUT>]

  --store <V4-ROOT>       directory NAMED 'cold-storage-v4'
  --keep-file <FILE>      session ids to PROTECT, one per line; '#' comments.
                          Preferred over --keep: process arguments are visible
                          to every user on the machine.
  --keep <SESSION-ID>     one id to protect; repeatable. Exposes the id in argv.
  --keep-none             assert an EMPTY keep list deliberately. Without it an
                          empty effective set is refused, because an unset shell
                          variable would otherwise report the whole store as
                          unprotected and exit 0.
  --store-is-quiescent    REQUIRED. Asserts no server is writing this store.
                          This tool CANNOT verify it: session index writes are
                          guarded by a process-local mutex, not by the
                          cross-process store lock.
  --artifact <OUT>        write the exact manifest and block hashes, with
                          provenance, as evidence for a LATER deletion decision.

EXIT CODES:
  0  report complete; every named session matched an index
  1  report complete, but a named session matched nothing
  3  report INCOMPLETE — figures withheld or partial; do not act on this run
  2  usage, validation, or I/O error

This report is EVIDENCE, not a grant. It says what nothing in THIS keep list
protects. It does not say anything is safe to delete.
"
    .to_string()
}

fn parse_args() -> Result<Args, String> {
    let mut store: Option<PathBuf> = None;
    let mut keep: Vec<String> = Vec::new();
    let (mut keep_none, mut quiescent) = (false, false);
    let mut artifact: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
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

    // `--store` is the BASE directory. `BlockColdStore` appends the v4 root
    // itself (`base_dir.join(V4_ROOT)`), so passing the v4 root here yields
    // `cold-storage-v4/cold-storage-v4`, which does not exist, which enumerates
    // nothing, which prints as a clean zero report — the one output an operator
    // reads as "nothing to do".
    //
    // An earlier revision of this validation demanded the opposite and thereby
    // GUARANTEED that failure. Caught by running it, not by reading it.
    let v4 = store.join(V4_ROOT_NAME);
    if !v4.is_dir() {
        return Err(format!(
            "--store must be the directory CONTAINING '{V4_ROOT_NAME}', not the v4 root \
             itself.\n\
             Looked for: {}\n\n\
             The store appends '{V4_ROOT_NAME}' internally. A wrong level enumerates \
             nothing and prints as a clean zero report.",
            v4.display()
        ));
    }
    for sub in ["manifests", "sessions", "blocks"] {
        let p = v4.join(sub);
        if !p.is_dir() {
            return Err(format!(
                "{} is not a directory — is this a v4 store?",
                p.display()
            ));
        }
    }
    if !quiescent {
        return Err("--store-is-quiescent is required.\n\n\
             This tool CANNOT verify quiescence: session index writes are guarded by a \
             process-local mutex in the serving process, not by the cross-process store \
             lock, so a separate binary cannot exclude them. A scan taken while the \
             server writes can classify a manifest a NAMED conversation depends on as \
             unprotected.\n\n\
             Stop the server, then re-run with --store-is-quiescent."
            .to_string());
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
            "{dupes} duplicate session id(s) in the keep list. Duplicates inflate the \
             named count while the matched count deduplicates, so the two disagree for \
             a reason that is not a finding. Remove them and re-run."
        ));
    }
    Ok(Args {
        store,
        keep,
        keep_none,
        artifact,
    })
}

/// Cheap contradiction check on the quiescence assertion.
///
/// A POSITIVE result refuses the run. A negative result proves nothing — the
/// server may be mid-restart, renamed, or on another host sharing the volume.
/// This catches the obvious mistake; it does not establish the invariant.
///
/// FAILS CLOSED. `pgrep` exits 1 for "no match", which is a real negative; any
/// other failure means the check could not run, and a check that could not run
/// must not be reported as a check that found nothing. That is the one shape
/// where a guard's failure is indistinguishable from its pass.
enum Quiescence {
    NoServerFound,
    ServerRunning,
    CouldNotCheck(String),
}

fn probe_for_live_server() -> Quiescence {
    match std::process::Command::new("/usr/bin/pgrep")
        .arg("-x")
        .arg("mlxcel-server")
        .output()
    {
        Err(e) => Quiescence::CouldNotCheck(format!("could not run pgrep: {e}")),
        Ok(o) => match o.status.code() {
            Some(0) if !o.stdout.is_empty() => Quiescence::ServerRunning,
            // pgrep's documented "no processes matched".
            Some(1) => Quiescence::NoServerFound,
            other => Quiescence::CouldNotCheck(format!(
                "pgrep exited {other:?}, which is neither a match nor a clean no-match"
            )),
        },
    }
}

pub struct Buckets {
    pub keep: Vec<[u8; 32]>,
    pub unprotected: Vec<[u8; 32]>,
    pub unattributed: Vec<[u8; 32]>,
    pub matched_digests: HashSet<[u8; 32]>,
}

/// The classification, as a pure function over already-read state.
///
/// Separated from `main` so the one load-bearing decision in this tool is
/// reachable by a test at all.
fn classify(
    indexes: &[([u8; 32], Vec<[u8; 32]>)],
    manifest_hashes: &[[u8; 32]],
    keep_digests: &HashSet<[u8; 32]>,
) -> Buckets {
    let mut referenced_by_kept: HashSet<[u8; 32]> = HashSet::new();
    let mut referenced_by_unkept: HashSet<[u8; 32]> = HashSet::new();
    let mut matched_digests: HashSet<[u8; 32]> = HashSet::new();

    for (digest, manifests) in indexes {
        let kept = keep_digests.contains(digest);
        if kept {
            matched_digests.insert(*digest);
        }
        for m in manifests {
            if kept {
                referenced_by_kept.insert(*m);
            } else {
                referenced_by_unkept.insert(*m);
            }
        }
    }

    let (mut keep, mut unprotected, mut unattributed) = (Vec::new(), Vec::new(), Vec::new());
    for h in manifest_hashes {
        // ORDER IS THE SAFETY PROPERTY. Kept wins, so a manifest shared between
        // a protected conversation and an unprotected one stays protected.
        // Reversing these arms would report shared prefixes as unprotected —
        // and shared prefixes are the common case, because that is what content
        // addressing is for.
        if referenced_by_kept.contains(h) {
            keep.push(*h);
        } else if referenced_by_unkept.contains(h) {
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

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    match probe_for_live_server() {
        Quiescence::ServerRunning => {
            eprintln!(
                "REFUSING: an mlxcel-server process is running. --store-is-quiescent was \
                 asserted and is contradicted. Stop the server and re-run.\n\n\
                 (The reverse does not hold: no detected server is NOT proof of \
                 quiescence.)"
            );
            return ExitCode::from(2);
        }
        Quiescence::CouldNotCheck(why) => {
            eprintln!(
                "REFUSING: the contradiction check on --store-is-quiescent could not \
                 run ({why}). A check that did not run is not a check that found \
                 nothing."
            );
            return ExitCode::from(2);
        }
        Quiescence::NoServerFound => {}
    }

    // The runtime fingerprint participates in manifest ADDRESSING, not in
    // reading one by hash. A report that enumerates and reads by hash never
    // consults it.
    let store = BlockColdStore::new(args.store.clone(), [0u8; 32]);

    let raw = match store.enumerate_session_indexes() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot enumerate session indexes: {e}");
            return ExitCode::from(2);
        }
    };

    // PATH/HEADER AGREEMENT, checked rather than trusted. `enumerate_session_indexes`
    // returns what each file CLAIMS. `session_associations(key)` can compare that
    // claim against an expected digest; an enumeration has no expectation, so
    // the filename stem is the only independent witness. A file placed or
    // symlinked under one session's name while claiming another would otherwise
    // expose the first session's manifests as unprotected. A mismatch fails the
    // whole report and is NEVER converted to absence.
    let sessions_dir = args.store.join("sessions");
    let mut indexes: Vec<([u8; 32], Vec<[u8; 32]>)> = Vec::new();
    for entry in raw {
        let Some(digest) = entry.key_digest else {
            eprintln!(
                "REPORT INCOMPLETE: a session index vanished between the directory scan \
                 and the read. The store is not quiescent, or the scan raced a writer."
            );
            return ExitCode::from(3);
        };
        let expected = sessions_dir.join(format!("{}.idx", hex32(&digest)));
        match expected.symlink_metadata() {
            Ok(m) if m.file_type().is_file() => {}
            Ok(_) => {
                eprintln!(
                    "REFUSING: {} is a symlink or not a regular file.",
                    expected.display()
                );
                return ExitCode::from(2);
            }
            Err(_) => {
                eprintln!(
                    "REFUSING: an index claims digest {} but no regular file of that \
                     name exists. The filename stem and the header disagree, so the \
                     file's session identity cannot be established.",
                    hex32(&digest)
                );
                return ExitCode::from(2);
            }
        }
        indexes.push((
            digest,
            entry.associations.iter().map(|a| a.manifest_hash).collect(),
        ));
    }

    let manifest_hashes = match store.enumerate_manifest_hashes() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot enumerate manifests: {e}");
            return ExitCode::from(2);
        }
    };

    let keep_digests: HashSet<[u8; 32]> = args
        .keep
        .iter()
        .map(|k| BlockColdStore::key_digest_of(k))
        .collect();
    let b = classify(&indexes, &manifest_hashes, &keep_digests);

    // Block reachability against a HYPOTHETICAL root set. Nothing is marked,
    // tombstoned or written. UNATTRIBUTED joins the protected side.
    let mut protected_blocks: HashSet<[u8; 32]> = HashSet::new();
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
    gather(&b.keep, &mut protected_blocks);
    // UNATTRIBUTED joins the PROTECTED side: a manifest whose association was
    // merely lost may belong to a live conversation, so its blocks must not
    // become candidates.
    gather(&b.unattributed, &mut protected_blocks);
    gather(&b.unprotected, &mut unprotected_blocks);
    drop(gather);
    // An unreadable manifest cannot prove what it references, and an unknown
    // root can overlap any candidate. Withhold every figure rather than
    // approximate one — a qualified number still gets read as a number.
    if unreadable > 0 {
        eprintln!(
            "REPORT INCOMPLETE: {unreadable} manifest(s) unreadable, so reachability \
             cannot be established. No block or byte figure is produced. Do not act \
             on this run."
        );
        return ExitCode::from(3);
    }

    let mut candidate: Vec<[u8; 32]> = unprotected_blocks
        .difference(&protected_blocks)
        .copied()
        .collect();
    candidate.sort_unstable();

    let blocks_dir = args.store.join("blocks");
    let (mut bytes, mut unsized_n) = (0u64, 0usize);
    for blk in &candidate {
        match dir_bytes(&blocks_dir.join(hex32(blk))) {
            Some(n) => bytes = bytes.saturating_add(n),
            None => unsized_n += 1,
        }
    }

    let unmatched: Vec<(usize, &String)> = args
        .keep
        .iter()
        .enumerate()
        .filter(|(_, k)| !b.matched_digests.contains(&BlockColdStore::key_digest_of(k)))
        .collect();

    println!("cold store    {}", args.store.display());
    println!("quiescence    OPERATOR-ASSERTED, not verified by this tool");
    println!(
        "sessions      {} indexes; {} named{}, {} matched",
        indexes.len(),
        args.keep.len(),
        if args.keep_none { " (--keep-none)" } else { "" },
        b.matched_digests.len()
    );
    println!();
    println!("KEEP          {:>6} manifests", b.keep.len());
    println!(
        "UNPROTECTED   {:>6} manifests   → {} blocks referenced by no KEEP or \
         UNATTRIBUTED manifest",
        b.unprotected.len(),
        candidate.len()
    );
    println!(
        "UNATTRIBUTED  {:>6} manifests   [protected; never proposed]",
        b.unattributed.len()
    );
    println!();
    if unsized_n == 0 {
        println!("those blocks occupy {:.2} GiB on disk", gib(bytes));
    } else {
        println!("SIZE INCOMPLETE: {unsized_n} block(s) could not be sized; no total given");
    }
    println!();
    println!(
        "UNPROTECTED means: nothing in THIS keep list references it. It does NOT mean \
         safe to delete.\nAn unnamed session may be active, resumable, or simply \
         forgotten."
    );

    if let Some(path) = &args.artifact {
        // The exact objects, so a later deletion decision can name what a human
        // actually reviewed. Aggregate counts cannot authorize specific objects,
        // and a later tool recomputing the set would not inherit this review.
        let mut out = String::new();
        out.push_str("# cold-store-keeplist artifact — EVIDENCE, NOT A DELETION GRANT\n");
        out.push_str(&format!("store\t{}\n", args.store.display()));
        out.push_str(&format!(
            "code_commit\t{}\n",
            option_env!("MLXCEL_GIT_SHA").unwrap_or("unknown")
        ));
        out.push_str("quiescence\toperator-asserted\n");
        out.push_str(&format!("sessions_enumerated\t{}\n", indexes.len()));
        out.push_str(&format!("keep_named\t{}\n", args.keep.len()));
        for k in &args.keep {
            out.push_str(&format!("keep_label\t{}\n", short_label(k)));
        }
        out.push_str(&format!(
            "completeness\t{}\n",
            if unsized_n == 0 {
                "complete"
            } else {
                "size-incomplete"
            }
        ));
        for h in &b.unprotected {
            out.push_str(&format!("unprotected_manifest\t{}\n", hex32(h)));
        }
        for blk in &candidate {
            out.push_str(&format!("candidate_block\t{}\n", hex32(blk)));
        }
        if let Err(e) = std::fs::write(path, out) {
            eprintln!("cannot write artifact {}: {e}", path.display());
            return ExitCode::from(2);
        }
        println!("\nartifact      {}", path.display());
    }

    if !unmatched.is_empty() {
        println!();
        println!("UNMATCHED — these protected nothing:");
        for (i, k) in &unmatched {
            println!("  keep-list entry {}  (digest {})", i + 1, short_label(k));
        }
        return ExitCode::from(1);
    }
    if unsized_n > 0 {
        return ExitCode::from(3);
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(n: u8) -> [u8; 32] {
        [n; 32]
    }

    fn idx(d: u8, manifests: &[u8]) -> ([u8; 32], Vec<[u8; 32]>) {
        (h(d), manifests.iter().map(|m| h(*m)).collect())
    }

    /// THE SAFETY PROPERTY. A manifest shared between a kept session and an
    /// unkept one is KEPT. Shared prefixes are the common case, so reversing
    /// the arms would report most of the store as unprotected.
    ///
    /// MUTATION: swap the two arms in `classify` and this goes red.
    #[test]
    fn shared_manifest_is_protected_by_any_keeper() {
        let indexes = vec![idx(10, &[1, 2]), idx(20, &[1, 3])];
        let keep: HashSet<[u8; 32]> = [h(10)].into_iter().collect();
        let b = classify(&indexes, &[h(1), h(2), h(3)], &keep);
        assert!(b.keep.contains(&h(1)), "shared manifest must be KEPT");
        assert_eq!(b.unprotected, vec![h(3)]);
        assert!(b.unattributed.is_empty());
    }

    /// A manifest referenced by NO index is UNATTRIBUTED, never UNPROTECTED.
    ///
    /// MUTATION: change the final `else` arm to push into `unprotected`.
    #[test]
    fn unreferenced_manifest_is_unattributed() {
        let b = classify(&[idx(20, &[1])], &[h(1), h(99)], &HashSet::new());
        assert_eq!(b.unprotected, vec![h(1)]);
        assert_eq!(b.unattributed, vec![h(99)]);
    }

    /// UNATTRIBUTED must never leak into the candidate side. `main` unions its
    /// blocks with the protected set; this pins the bucket that drives it.
    #[test]
    fn unattributed_is_never_a_candidate_bucket() {
        let b = classify(&[idx(20, &[1])], &[h(1), h(99)], &HashSet::new());
        assert!(b.unattributed.contains(&h(99)));
        assert!(!b.unprotected.contains(&h(99)));
    }

    /// A keep-digest naming no index matches nothing, and the caller must see
    /// it. A typo'd id protects nothing while looking like it did.
    #[test]
    fn unmatched_keep_digest_is_reported_as_unmatched() {
        let typo = h(77);
        let keep: HashSet<[u8; 32]> = [typo].into_iter().collect();
        let b = classify(&[idx(20, &[1])], &[h(1)], &keep);
        assert!(!b.matched_digests.contains(&typo));
        assert_eq!(b.unprotected, vec![h(1)]);
    }

    /// Every manifest lands in exactly one bucket. One that fell out of all
    /// three would be invisible — neither protected nor listed nor flagged —
    /// which is the one outcome no reader could detect.
    #[test]
    fn buckets_partition_the_manifest_set_exactly() {
        let all = [h(1), h(2), h(3), h(4), h(5)];
        let indexes = vec![idx(10, &[1]), idx(20, &[2, 3])];
        let keep: HashSet<[u8; 32]> = [h(10)].into_iter().collect();
        let b = classify(&indexes, &all, &keep);
        assert_eq!(
            b.keep.len() + b.unprotected.len() + b.unattributed.len(),
            all.len()
        );
        let union: HashSet<[u8; 32]> = b
            .keep
            .iter()
            .chain(&b.unprotected)
            .chain(&b.unattributed)
            .copied()
            .collect();
        assert_eq!(union.len(), all.len(), "no manifest may appear twice");
    }

    /// Output ordering is deterministic, so two runs over the same state give
    /// byte-identical artifacts. An artifact that reorders cannot be compared
    /// against the one a human reviewed.
    #[test]
    fn bucket_output_is_sorted_for_artifact_determinism() {
        let b = classify(&[idx(20, &[5, 1, 3])], &[h(5), h(1), h(3)], &HashSet::new());
        assert_eq!(b.unprotected, vec![h(1), h(3), h(5)]);
    }

    #[test]
    fn key_digest_is_stable_and_distinguishing() {
        let a = BlockColdStore::key_digest_of("session-a");
        assert_eq!(a, BlockColdStore::key_digest_of("session-a"));
        assert_ne!(a, BlockColdStore::key_digest_of("session-b"));
    }

    /// The short label must not reveal the id and must distinguish ids.
    #[test]
    fn short_label_is_non_reversible_and_distinguishing() {
        let a = short_label("session-a");
        assert_eq!(a.len(), 12);
        assert!(!a.contains("session"));
        assert_ne!(a, short_label("session-b"));
    }
}
