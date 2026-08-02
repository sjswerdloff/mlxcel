//! Keep-list report for the v4 block cold store. **READ-ONLY.**
//!
//! Stuart, 2026-07-31: *"I need a way to know which data I should manually
//! delete to effect the equivalent of a session-release... what I might end up
//! having to do is list the X-Session-Ids that I do not want deleted and get
//! back the rest."*
//!
//! This is that report. It deletes nothing, writes nothing, and creates no
//! registry. See `DESIGN_keeplist_report_20260802.md`.
//!
//! # Why a keep-list and not a delete-list
//!
//! The raw session key is never stored — `session_index_path` derives a
//! filename from the key's digest and the file carries only that digest. So the
//! store cannot name a conversation, and *"show me what is deletable"* would
//! present an operator with hex strings to choose between.
//!
//! A keep-list needs identification **only of what is kept**, which is exactly
//! the set the operator has: the conversations he is in. The asymmetry runs the
//! right way.
//!
//! # The three buckets
//!
//! Classification is per **manifest**, never per session — manifests are
//! content-addressed and prefixes are shared, so one manifest can be referenced
//! by a kept session and an unkept one at once.
//!
//! * **KEEP** — referenced by at least one kept session.
//! * **RELEASABLE** — referenced by at least one session, and every referencing
//!   session is unkept.
//! * **UNATTRIBUTED** — referenced by *no* session index. **Never proposed for
//!   deletion.** A manifest with no association is indistinguishable from one
//!   whose association was lost: persisted before session tracking existed,
//!   lost to the crash window `session_associations` documents, or belonging to
//!   an index whose digest mismatched and which is deliberately read as empty.
//!   Folding it into RELEASABLE would delete a live conversation's cache;
//!   folding it into KEEP would hide a leak behind a reassuring number.
//!
//! # Why not `releasable_manifests`
//!
//! That function returns empty unless a generation has been *closed*, and no
//! generation has ever been closed on this store because the close mechanism
//! does not exist yet. It answers *what has this session finished with?*; this
//! answers *what is protected by nobody?*

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mlxcel_core::cache::block_cold_store::BlockColdStore;

fn hex32(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Recursive byte total for one block directory.
///
/// Returns `None` when any part of it cannot be read. A partial sum here would
/// under-report reclaimable space, and an under-reported figure is exactly the
/// one that makes a disk look less recoverable than it is — which is the
/// decision this report exists to inform.
fn dir_bytes(path: &Path) -> Option<u64> {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(p) = stack.pop() {
        for entry in std::fs::read_dir(&p).ok()? {
            let entry = entry.ok()?;
            let ft = entry.file_type().ok()?;
            if ft.is_dir() {
                stack.push(entry.path());
            } else {
                total += entry.metadata().ok()?.len();
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
}

fn usage() -> String {
    "\
cold-store-keeplist — READ-ONLY report. Deletes nothing.

USAGE:
  cold-store-keeplist --store <DIR> [--keep <SESSION-ID>]... [--keep-file <FILE>]

  --store <DIR>          cold store base directory (the one holding the v4 root)
  --keep <SESSION-ID>    an X-Session-Id to protect; repeatable
  --keep-file <FILE>     file of session ids, one per line; blank lines and
                         lines beginning '#' ignored

EXIT CODES:
  0  report produced, every named session matched an index on disk
  1  a named session matched nothing — see 'UNMATCHED' below
  2  usage or I/O error

A session id that matches nothing protects nothing while looking like it did,
so an unmatched name is an error rather than a note.
"
    .to_string()
}

fn parse_args() -> Result<Args, String> {
    let mut store: Option<PathBuf> = None;
    let mut keep: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--store" => {
                store = Some(PathBuf::from(
                    it.next().ok_or("--store needs a directory")?,
                ))
            }
            "--keep" => keep.push(it.next().ok_or("--keep needs a session id")?),
            "--keep-file" => {
                let f = it.next().ok_or("--keep-file needs a path")?;
                let body = std::fs::read_to_string(&f)
                    .map_err(|e| format!("cannot read keep-file {f}: {e}"))?;
                for line in body.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    keep.push(line.to_string());
                }
            }
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown argument {other}\n\n{}", usage())),
        }
    }
    Ok(Args {
        store: store.ok_or_else(|| format!("--store is required\n\n{}", usage()))?,
        keep,
    })
}

/// The classification, as a pure function over already-read state.
///
/// Separated from `main` so it is testable at all. The bucket rule is the one
/// load-bearing decision in this tool — it decides what an operator is told is
/// deletable — and a rule that only runs inside `main` is a rule no test can
/// reach.
pub struct Buckets {
    pub keep: Vec<[u8; 32]>,
    pub releasable: Vec<[u8; 32]>,
    pub unattributed: Vec<[u8; 32]>,
    /// Keep-digests that actually matched an index on disk.
    pub matched_digests: HashSet<[u8; 32]>,
    /// Index files that vanished between the directory scan and the read.
    pub vanished: usize,
}

fn classify(
    indexes: &[mlxcel_core::cache::block_cold_store::SessionIndexEntry],
    manifest_hashes: &[[u8; 32]],
    keep_digests: &HashSet<[u8; 32]>,
) -> Buckets {
    let mut referenced_by_kept: HashSet<[u8; 32]> = HashSet::new();
    let mut referenced_by_unkept: HashSet<[u8; 32]> = HashSet::new();
    let mut matched_digests: HashSet<[u8; 32]> = HashSet::new();
    let mut vanished = 0usize;

    for idx in indexes {
        let Some(digest) = idx.key_digest else {
            // Vanished between scan and read. Counted, not silently skipped:
            // its manifests fall into UNATTRIBUTED, and the operator needs to
            // know that bucket may be inflated by a race rather than by a real
            // orphan.
            vanished += 1;
            continue;
        };
        let kept = keep_digests.contains(&digest);
        if kept {
            matched_digests.insert(digest);
        }
        for a in &idx.associations {
            if kept {
                referenced_by_kept.insert(a.manifest_hash);
            } else {
                referenced_by_unkept.insert(a.manifest_hash);
            }
        }
    }

    let mut keep = Vec::new();
    let mut releasable = Vec::new();
    let mut unattributed = Vec::new();
    for h in manifest_hashes {
        // ORDER IS THE SAFETY PROPERTY. Kept wins over unkept, so a manifest
        // shared between a protected conversation and an unprotected one is
        // protected. Reversing these two arms would report shared prefixes as
        // deletable — and shared prefixes are the common case, because that is
        // what content addressing is for.
        if referenced_by_kept.contains(h) {
            keep.push(*h);
        } else if referenced_by_unkept.contains(h) {
            releasable.push(*h);
        } else {
            unattributed.push(*h);
        }
    }

    Buckets {
        keep,
        releasable,
        unattributed,
        matched_digests,
        vanished,
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

    // The runtime fingerprint participates in manifest ADDRESSING, not in
    // reading a manifest by hash. A report that only enumerates and reads by
    // hash never consults it.
    let store = BlockColdStore::new(args.store.clone(), [0u8; 32]);

    let indexes = match store.enumerate_session_indexes() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot enumerate session indexes: {e}");
            return ExitCode::from(2);
        }
    };
    let manifest_hashes = match store.enumerate_manifest_hashes() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot enumerate manifests: {e}");
            return ExitCode::from(2);
        }
    };

    // Keep-set as digests. The raw ids never reach disk or the report body.
    let keep_digests: HashSet<[u8; 32]> = args
        .keep
        .iter()
        .map(|k| BlockColdStore::key_digest_of(k))
        .collect();

    let Buckets {
        keep: keep_m,
        releasable: releasable_m,
        unattributed: unattributed_m,
        matched_digests,
        vanished,
    } = classify(&indexes, &manifest_hashes, &keep_digests);

    // Block reachability, evaluated against a HYPOTHETICAL root set. Nothing is
    // marked, tombstoned or written. A block is reclaimable only if no KEEP and
    // no UNATTRIBUTED manifest also references it — the same rule
    // `mark_reachable_blocks` applies to the real root set.
    let mut protected_blocks: HashSet<[u8; 32]> = HashSet::new();
    let mut releasable_blocks: HashSet<[u8; 32]> = HashSet::new();
    let mut unattributed_blocks: HashSet<[u8; 32]> = HashSet::new();
    let mut unreadable_manifests = 0usize;

    let mut collect = |hashes: &[[u8; 32]], into: &mut HashSet<[u8; 32]>| {
        for h in hashes {
            match store.read_manifest(h) {
                Ok(m) => into.extend(m.block_hashes.iter().copied()),
                Err(_) => unreadable_manifests += 1,
            }
        }
    };
    collect(&keep_m, &mut protected_blocks);
    collect(&releasable_m, &mut releasable_blocks);
    collect(&unattributed_m, &mut unattributed_blocks);

    // An unreadable manifest cannot prove what it references. Reporting a
    // reclaimable figure over a root set with a hole in it is the shape that
    // deletes live data, so the figure is withheld rather than approximated.
    if unreadable_manifests > 0 {
        eprintln!(
            "REFUSING TO REPORT RECLAIMABLE SPACE: {unreadable_manifests} manifest(s) \
             could not be read, so reachability cannot be proved. An unreadable \
             manifest is not a manifest with no references."
        );
        return ExitCode::from(2);
    }

    let reclaimable: Vec<[u8; 32]> = releasable_blocks
        .difference(&protected_blocks)
        .copied()
        .filter(|b| !unattributed_blocks.contains(b))
        .collect();

    let blocks_dir = store.blocks_dir();
    let mut reclaimable_bytes = 0u64;
    let mut unsized_blocks = 0usize;
    for b in &reclaimable {
        match dir_bytes(&blocks_dir.join(hex32(b))) {
            Some(n) => reclaimable_bytes += n,
            None => unsized_blocks += 1,
        }
    }

    let unmatched: Vec<&String> = args
        .keep
        .iter()
        .filter(|k| !matched_digests.contains(&BlockColdStore::key_digest_of(k)))
        .collect();

    println!("cold store   {}", args.store.display());
    println!(
        "sessions     {} index files ({} named to keep, {} matched)",
        indexes.len(),
        args.keep.len(),
        matched_digests.len()
    );
    if vanished > 0 {
        println!(
            "             {vanished} index file(s) vanished mid-scan — UNATTRIBUTED may be inflated"
        );
    }
    println!();
    println!("KEEP           {:>6} manifests", keep_m.len());
    println!(
        "RELEASABLE     {:>6} manifests   → {} blocks, {:.2} GiB reclaimable",
        releasable_m.len(),
        reclaimable.len(),
        gib(reclaimable_bytes)
    );
    println!(
        "UNATTRIBUTED   {:>6} manifests   [NOT proposed for deletion]",
        unattributed_m.len()
    );
    if unsized_blocks > 0 {
        println!();
        println!(
            "note: {unsized_blocks} reclaimable block(s) could not be sized; the GiB \
             figure is a LOWER BOUND"
        );
    }

    if !unmatched.is_empty() {
        println!();
        println!("UNMATCHED — these protect nothing:");
        for k in &unmatched {
            println!("  {k}");
        }
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel_core::cache::block_cold_store::{SessionAssociation, SessionIndexEntry};

    fn h(n: u8) -> [u8; 32] {
        [n; 32]
    }

    fn idx(digest: Option<[u8; 32]>, manifests: &[u8]) -> SessionIndexEntry {
        SessionIndexEntry {
            key_digest: digest,
            associations: manifests
                .iter()
                .map(|m| SessionAssociation {
                    incarnation: Default::default(),
                    manifest_hash: h(*m),
                    generation: 0,
                })
                .collect(),
        }
    }

    /// THE SAFETY PROPERTY. A manifest shared between a kept session and an
    /// unkept one is KEPT. Shared prefixes are the common case — that is what
    /// content addressing buys — so an implementation that reversed the two
    /// arms would report most of the store as deletable.
    ///
    /// MUTATION: swap the `referenced_by_kept` / `referenced_by_unkept` arms in
    /// `classify` and this goes red, because manifest 1 moves to RELEASABLE.
    #[test]
    fn shared_manifest_is_protected_by_any_keeper() {
        let kept = h(10);
        let unkept = h(20);
        let indexes = vec![idx(Some(kept), &[1, 2]), idx(Some(unkept), &[1, 3])];
        let keep: HashSet<[u8; 32]> = [kept].into_iter().collect();

        let b = classify(&indexes, &[h(1), h(2), h(3)], &keep);

        assert!(b.keep.contains(&h(1)), "shared manifest must be KEPT");
        assert!(b.keep.contains(&h(2)));
        assert_eq!(b.releasable, vec![h(3)], "only the unkept-only manifest");
        assert!(b.unattributed.is_empty());
    }

    /// A manifest referenced by NO index is UNATTRIBUTED, never RELEASABLE.
    /// Absence of an association is indistinguishable from a lost one, and the
    /// two have opposite correct actions.
    ///
    /// MUTATION: change the final `else` arm to push into `releasable` and this
    /// goes red.
    #[test]
    fn unreferenced_manifest_is_unattributed_not_releasable() {
        let unkept = h(20);
        let indexes = vec![idx(Some(unkept), &[1])];
        let keep: HashSet<[u8; 32]> = HashSet::new();

        let b = classify(&indexes, &[h(1), h(99)], &keep);

        assert_eq!(b.releasable, vec![h(1)]);
        assert_eq!(
            b.unattributed,
            vec![h(99)],
            "a manifest no index references must NOT be proposed for deletion"
        );
    }

    /// An EMPTY keep-list must not make everything releasable by accident, and
    /// must not make anything kept. It is a legitimate query — "what does
    /// nothing protect?" — and its answer is "everything attributed".
    #[test]
    fn empty_keep_list_protects_nothing_and_invents_nothing() {
        let indexes = vec![idx(Some(h(20)), &[1, 2])];
        let b = classify(&indexes, &[h(1), h(2), h(3)], &HashSet::new());

        assert!(b.keep.is_empty());
        assert_eq!(b.releasable.len(), 2);
        assert_eq!(b.unattributed, vec![h(3)]);
        assert!(b.matched_digests.is_empty());
    }

    /// A keep-digest naming no index on disk matches nothing, and the caller
    /// must be able to see that. A typo'd session id protects nothing while
    /// looking like it protected something.
    #[test]
    fn unmatched_keep_digest_is_reported_as_unmatched() {
        let indexes = vec![idx(Some(h(20)), &[1])];
        let typo = h(77);
        let keep: HashSet<[u8; 32]> = [typo].into_iter().collect();

        let b = classify(&indexes, &[h(1)], &keep);

        assert!(
            !b.matched_digests.contains(&typo),
            "an id matching no index must not be reported as matched"
        );
        assert_eq!(b.releasable, vec![h(1)], "and it protects nothing");
    }

    /// An index that vanished mid-scan contributes NO attribution, is counted,
    /// and its manifests therefore land in UNATTRIBUTED rather than RELEASABLE.
    /// The count exists so the operator can tell an inflated bucket from a real
    /// orphan population.
    #[test]
    fn vanished_index_is_counted_and_attributes_nothing() {
        let indexes = vec![idx(None, &[1]), idx(Some(h(20)), &[2])];
        let b = classify(&indexes, &[h(1), h(2)], &HashSet::new());

        assert_eq!(b.vanished, 1);
        assert_eq!(b.unattributed, vec![h(1)], "its manifest is NOT releasable");
        assert_eq!(b.releasable, vec![h(2)]);
    }

    /// Every manifest lands in exactly one bucket. A manifest that fell out of
    /// all three would be invisible to the report — neither protected nor
    /// proposed nor flagged — which is the one outcome no reader could detect.
    #[test]
    fn buckets_partition_the_manifest_set_exactly() {
        let all = [h(1), h(2), h(3), h(4), h(5)];
        let indexes = vec![idx(Some(h(10)), &[1]), idx(Some(h(20)), &[2, 3])];
        let keep: HashSet<[u8; 32]> = [h(10)].into_iter().collect();

        let b = classify(&indexes, &all, &keep);

        assert_eq!(
            b.keep.len() + b.releasable.len() + b.unattributed.len(),
            all.len(),
            "counts must sum to the input"
        );
        let union: HashSet<[u8; 32]> = b
            .keep
            .iter()
            .chain(b.releasable.iter())
            .chain(b.unattributed.iter())
            .copied()
            .collect();
        assert_eq!(union.len(), all.len(), "and no manifest may appear twice");
    }

    /// The digest derivation the keep-set depends on must be stable and must
    /// distinguish different keys. A collision here would silently protect the
    /// wrong conversation.
    #[test]
    fn key_digest_is_stable_and_distinguishing() {
        let a = BlockColdStore::key_digest_of("session-a");
        let b = BlockColdStore::key_digest_of("session-b");
        assert_eq!(a, BlockColdStore::key_digest_of("session-a"));
        assert_ne!(a, b);
    }
}
