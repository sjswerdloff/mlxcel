// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Opt-in normalization for volatile Claude Code top-level system text.

use std::fmt;

const TASK_NUDGE: &str = "The task tools haven't been used recently. If you're working on tasks that would benefit from tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status (set to in_progress when starting, completed when done). Also consider cleaning up the task list if it has become stale. Only use these if relevant to the current work. This is just a gentle reminder - ignore if not applicable.";
const TASK_DUMP_HEADER: &str = "Here are the existing tasks:";
const TOOLS_HEADING: &str = "# Tools";
const DATE_PREFIX: &str = "The date has changed. Today's date is now ";
const DATE_SUFFIX: &str =
    ". DO NOT mention this to the user explicitly because they are already aware.";
const STABLE_PREFIX_V1_CACHE_TAG: &str = "claude-code-prompt-normalization:stable-prefix-v1";
const MAX_BLANK_LINES_BEFORE_TOOLS: usize = 2;
const TASK_DUMP_BLANK_LINES_AFTER_NUDGE: usize = 2;
const TASK_DUMP_BLANK_LINES_AFTER_HEADER: usize = 1;

/// Claude Code prompt normalization policy applied before chat rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ClaudeCodePromptNormalization {
    /// Preserve the request text exactly (apart from legacy billing-header stripping).
    #[default]
    Off,
    /// Remove the bounded volatile reminders recognized by version 1.
    StablePrefixV1,
}

impl fmt::Display for ClaudeCodePromptNormalization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::StablePrefixV1 => "stable-prefix-v1",
        })
    }
}

impl ClaudeCodePromptNormalization {
    /// Namespace an Anthropic prompt-cache template signature by this policy.
    ///
    /// `Off` deliberately preserves the historical signature. The enabled
    /// policy adds a stable version tag so cache buckets cannot cross a future
    /// normalization-policy transition even if the resulting token ids match.
    pub fn namespace_template_sig(self, template_sig: &str) -> String {
        match self {
            Self::Off => template_sig.to_string(),
            Self::StablePrefixV1 => format!("{template_sig}:{STABLE_PREFIX_V1_CACHE_TAG}"),
        }
    }
}

/// Counts of recognized blocks removed from one top-level system prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NormalizationEvents {
    pub task_nudges: usize,
    pub date_reminders: usize,
}

impl NormalizationEvents {
    pub fn total(self) -> usize {
        self.task_nudges + self.date_reminders
    }
}

/// Normalize top-level Anthropic system text according to `policy`.
pub fn normalize_top_level_system_text(
    text: &str,
    policy: ClaudeCodePromptNormalization,
) -> (String, NormalizationEvents) {
    if policy == ClaudeCodePromptNormalization::Off {
        return (text.to_string(), NormalizationEvents::default());
    }

    // The observed Claude Code order is task nudge, date reminder, then
    // `# Tools`. Removing the date first exposes the task nudge's structural
    // heading anchor without weakening either recognizer.
    let (text, date_reminders) = remove_date_reminder(text);
    let (text, task_nudges) = remove_task_nudge(&text);
    (
        text,
        NormalizationEvents {
            task_nudges,
            date_reminders,
        },
    )
}

#[derive(Debug, Clone, Copy)]
struct Line<'a> {
    start: usize,
    content: &'a str,
}

fn lines(text: &str) -> Vec<Line<'_>> {
    let mut out = Vec::new();
    let mut start = 0;
    for raw in text.split_inclusive('\n') {
        let without_lf = raw.strip_suffix('\n').unwrap_or(raw);
        let content = without_lf.strip_suffix('\r').unwrap_or(without_lf);
        let end = start + raw.len();
        out.push(Line { start, content });
        start = end;
    }
    if text.is_empty() {
        return out;
    }
    if start < text.len() {
        let raw = &text[start..];
        out.push(Line {
            start,
            content: raw.strip_suffix('\r').unwrap_or(raw),
        });
    }
    out
}

fn remove_task_nudge(text: &str) -> (String, usize) {
    let prompt_lines = lines(text);
    let candidates: Vec<usize> = prompt_lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (line.content == TASK_NUDGE).then_some(index))
        .collect();
    if candidates.len() != 1 {
        return (text.to_string(), 0);
    }

    let start_index = candidates[0];
    let Some(heading_index) = unique_tools_heading_index(&prompt_lines) else {
        return (text.to_string(), 0);
    };
    if !is_followed_by_tools_heading(&prompt_lines, start_index, heading_index)
        && !has_valid_task_dump(&prompt_lines, start_index, heading_index)
    {
        return (text.to_string(), 0);
    }

    let start = prompt_lines[start_index].start;
    let end = prompt_lines[heading_index].start;
    (remove_range(text, start, end), 1)
}

fn remove_date_reminder(text: &str) -> (String, usize) {
    let prompt_lines = lines(text);
    let candidates: Vec<usize> = prompt_lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| is_exact_date_reminder(line.content).then_some(index))
        .collect();
    if candidates.len() != 1 {
        return (text.to_string(), 0);
    }

    let index = candidates[0];
    let Some(heading_index) = unique_tools_heading_index(&prompt_lines) else {
        return (text.to_string(), 0);
    };
    if !is_followed_by_tools_heading(&prompt_lines, index, heading_index) {
        return (text.to_string(), 0);
    }

    let start = prompt_lines[index].start;
    let end = prompt_lines[heading_index].start;
    (remove_range(text, start, end), 1)
}

fn unique_tools_heading_index(prompt_lines: &[Line<'_>]) -> Option<usize> {
    let mut headings = prompt_lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (line.content == TOOLS_HEADING).then_some(index));
    let heading = headings.next()?;
    headings.next().is_none().then_some(heading)
}

fn is_followed_by_tools_heading(
    prompt_lines: &[Line<'_>],
    candidate_index: usize,
    heading_index: usize,
) -> bool {
    let Some(blank_lines) = heading_index.checked_sub(candidate_index + 1) else {
        return false;
    };
    blank_lines <= MAX_BLANK_LINES_BEFORE_TOOLS
        && prompt_lines[candidate_index + 1..heading_index]
            .iter()
            .all(|line| line.content.is_empty())
}

fn has_valid_task_dump(
    prompt_lines: &[Line<'_>],
    nudge_index: usize,
    heading_index: usize,
) -> bool {
    if heading_index <= nudge_index {
        return false;
    }
    let between = &prompt_lines[nudge_index + 1..heading_index];
    let required_prefix_len =
        TASK_DUMP_BLANK_LINES_AFTER_NUDGE + 1 + TASK_DUMP_BLANK_LINES_AFTER_HEADER;
    if between.len() <= required_prefix_len
        || !between[..TASK_DUMP_BLANK_LINES_AFTER_NUDGE]
            .iter()
            .all(|line| line.content.is_empty())
        || between[TASK_DUMP_BLANK_LINES_AFTER_NUDGE].content != TASK_DUMP_HEADER
        || !between[TASK_DUMP_BLANK_LINES_AFTER_NUDGE + 1..required_prefix_len]
            .iter()
            .all(|line| line.content.is_empty())
    {
        return false;
    }

    let mut previous_id = 0;
    let mut task_count = 0;
    let mut index = required_prefix_len;
    while index < between.len() && !between[index].content.is_empty() {
        let Some(task_id) = parse_task_line(between[index].content) else {
            return false;
        };
        // Claude Code renders tasks in task-id order. Requiring strict
        // increase rejects duplicate and reordered/ambiguous dumps while still
        // allowing gaps left by tasks that no longer appear in the live list.
        if task_id <= previous_id {
            return false;
        }
        previous_id = task_id;
        task_count += 1;
        index += 1;
    }

    task_count > 0
        && between.len() - index <= MAX_BLANK_LINES_BEFORE_TOOLS
        && between[index..].iter().all(|line| line.content.is_empty())
}

fn parse_task_line(line: &str) -> Option<u64> {
    let rest = line.strip_prefix('#')?;
    let (id, rest) = rest.split_once(". [")?;
    if id.is_empty()
        || (id.len() > 1 && id.starts_with('0'))
        || !id.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let task_id = id.parse::<u64>().ok().filter(|id| *id > 0)?;
    let (status, subject) = rest.split_once("] ")?;
    // TaskList exposes these three persistent statuses. `deleted` is an
    // update command that removes a task, so it cannot appear in this dump.
    if !matches!(status, "pending" | "in_progress" | "completed")
        || subject.trim().is_empty()
        || subject.chars().any(char::is_control)
    {
        return None;
    }
    Some(task_id)
}

fn remove_range(text: &str, start: usize, end: usize) -> String {
    if start > end
        || end > text.len()
        || !text.is_char_boundary(start)
        || !text.is_char_boundary(end)
    {
        return text.to_string();
    }
    let mut normalized = String::with_capacity(text.len() - (end - start));
    normalized.push_str(&text[..start]);
    normalized.push_str(&text[end..]);
    normalized
}

fn is_exact_date_reminder(line: &str) -> bool {
    let Some(rest) = line.strip_prefix(DATE_PREFIX) else {
        return false;
    };
    let Some(date) = rest.strip_suffix(DATE_SUFFIX) else {
        return false;
    };
    is_valid_gregorian_iso_date(date)
}

fn is_valid_gregorian_iso_date(date: &str) -> bool {
    let bytes = date.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
    {
        return false;
    }

    let year = parse_decimal(&bytes[0..4]);
    let month = parse_decimal(&bytes[5..7]);
    let day = parse_decimal(&bytes[8..10]);
    if year == 0 {
        return false;
    }
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => return false,
    };
    (1..=max_day).contains(&day)
}

fn parse_decimal(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(0, |value, byte| value * 10 + u32::from(byte - b'0'))
}

fn is_leap_year(year: u32) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date_reminder(date: &str) -> String {
        format!("{DATE_PREFIX}{date}{DATE_SUFFIX}")
    }

    fn task_dump_prompt(task_lines: &str) -> String {
        format!("rules\n\n{TASK_NUDGE}\n\n\n{TASK_DUMP_HEADER}\n\n{task_lines}\n\n\n# Tools\nbody")
    }

    #[test]
    fn off_preserves_task_nudge_and_date_byte_for_byte() {
        let input = format!(
            "rules\n\n{TASK_NUDGE}\n\n# Tools\n\n{}\n",
            date_reminder("2026-07-20")
        );
        let (normalized, events) =
            normalize_top_level_system_text(&input, ClaudeCodePromptNormalization::Off);
        assert_eq!(normalized, input);
        assert_eq!(events.total(), 0);
    }

    #[test]
    fn known_task_nudge_present_and_absent_normalize_identically() {
        let present = format!("rules\n\n{TASK_NUDGE}\n\n# Tools\nbody");
        let absent = "rules\n\n# Tools\nbody";
        let (normalized, events) = normalize_top_level_system_text(
            &present,
            ClaudeCodePromptNormalization::StablePrefixV1,
        );
        assert_eq!(normalized, absent);
        assert_eq!(events.task_nudges, 1);
    }

    #[test]
    fn observed_task_then_date_order_normalizes_identically_to_neither() {
        let present = format!(
            "rules\n\n{TASK_NUDGE}\n\n{}\n\n# Tools\nbody",
            date_reminder("2026-07-20")
        );
        let absent = "rules\n\n# Tools\nbody";
        let (normalized, events) = normalize_top_level_system_text(
            &present,
            ClaudeCodePromptNormalization::StablePrefixV1,
        );
        assert_eq!(normalized, absent);
        assert_eq!(events.task_nudges, 1);
        assert_eq!(events.date_reminders, 1);
    }

    #[test]
    fn changing_valid_task_dumps_normalize_identically_to_no_block() {
        let first = task_dump_prompt(
            "#1. [pending] Inspect the live prompt\n#2. [in_progress] Implement normalization\n#3. [completed] Confirm framing",
        );
        let second = task_dump_prompt(
            "#2. [completed] Preserve the stable prefix\n#7. [pending] Run focused verification",
        );
        let absent = "rules\n\n# Tools\nbody";

        for present in [first, second] {
            let (normalized, events) = normalize_top_level_system_text(
                &present,
                ClaudeCodePromptNormalization::StablePrefixV1,
            );
            assert_eq!(normalized, absent);
            assert_eq!(events.task_nudges, 1);
        }
    }

    #[test]
    fn task_dump_then_date_composition_normalizes_identically_to_neither() {
        let present = format!(
            "rules\n\n{TASK_NUDGE}\n\n\n{TASK_DUMP_HEADER}\n\n#1. [in_progress] Normalize the prompt\n#4. [pending] Verify the result\n\n\n{}\n\n\n# Tools\nbody",
            date_reminder("2026-07-20")
        );
        let absent = "rules\n\n# Tools\nbody";
        let (normalized, events) = normalize_top_level_system_text(
            &present,
            ClaudeCodePromptNormalization::StablePrefixV1,
        );
        assert_eq!(normalized, absent);
        assert_eq!(events.task_nudges, 1);
        assert_eq!(events.date_reminders, 1);
    }

    #[test]
    fn malformed_or_ambiguous_task_dumps_fail_open() {
        let mut malformed = [
            "#1. [cancelled] Unknown status",
            "#1 [pending] Missing punctuation",
            "#1. [pending] Valid task\nextra prose",
            "#0. [pending] Zero task id",
            "#01. [pending] Leading-zero task id",
            "#1. [pending] ",
            "#1. [pending] First\n#1. [completed] Duplicate id",
            "#2. [pending] Later\n#1. [completed] Reordered id",
            "<system-reminder>\n#1. [pending] Wrapped task\n</system-reminder>",
        ]
        .map(task_dump_prompt)
        .to_vec();
        malformed.push(format!(
            "rules\n\n{TASK_NUDGE}\n\n\n{TASK_DUMP_HEADER}\n\n# Tools\nbody"
        ));

        for present in malformed {
            let (normalized, events) = normalize_top_level_system_text(
                &present,
                ClaudeCodePromptNormalization::StablePrefixV1,
            );
            assert_eq!(normalized, present, "unexpected task-dump match");
            assert_eq!(events.task_nudges, 0);
        }
    }

    #[test]
    fn different_valid_dates_normalize_identically() {
        let first = format!("rules\n\n{}\n\n# Tools", date_reminder("2024-02-29"));
        let second = format!("rules\n\n{}\n\n# Tools", date_reminder("2026-07-20"));
        let policy = ClaudeCodePromptNormalization::StablePrefixV1;
        assert_eq!(
            normalize_top_level_system_text(&first, policy).0,
            normalize_top_level_system_text(&second, policy).0
        );
    }

    #[test]
    fn exact_date_elsewhere_in_top_level_system_text_is_preserved() {
        let input = format!(
            "rules\n{}\nordinary system content\n# Tools",
            date_reminder("2026-07-20")
        );
        let (normalized, events) =
            normalize_top_level_system_text(&input, ClaudeCodePromptNormalization::StablePrefixV1);
        assert_eq!(normalized, input);
        assert_eq!(events.date_reminders, 0);
    }

    #[test]
    fn changed_or_ambiguous_tools_heading_keeps_exact_date() {
        for input in [
            format!("{}\n\n# Tooling", date_reminder("2026-07-20")),
            format!(
                "{}\n\n# Tools\nordinary system content\n# Tools",
                date_reminder("2026-07-20")
            ),
        ] {
            let (normalized, events) = normalize_top_level_system_text(
                &input,
                ClaudeCodePromptNormalization::StablePrefixV1,
            );
            assert_eq!(normalized, input);
            assert_eq!(events.date_reminders, 0);
        }
    }

    #[test]
    fn invalid_dates_and_malformed_prose_fail_open() {
        for input in [
            date_reminder("2023-02-29"),
            date_reminder("2024-13-01"),
            date_reminder("2024-01-00"),
            date_reminder("2024-1-01"),
            "The date changed. Today's date is now 2026-07-20. DO NOT mention this to the user explicitly because they are already aware.".to_string(),
        ] {
            let (normalized, events) = normalize_top_level_system_text(
                &input,
                ClaudeCodePromptNormalization::StablePrefixV1,
            );
            assert_eq!(normalized, input);
            assert_eq!(events.total(), 0);
        }
    }

    #[test]
    fn changed_task_structure_and_unknown_reminders_fail_open() {
        for input in [
            format!("{TASK_NUDGE}\n\n# Tooling"),
            format!("{TASK_NUDGE}\n\n\n\n# Tools"),
            "The todo tools haven't been used recently. This is just a gentle reminder - ignore if not applicable.\n\n# Tools".to_string(),
            "The task tools haven't been used recently. Changed middle prose. This is just a gentle reminder - ignore if not applicable.\n\n# Tools".to_string(),
        ] {
            let (normalized, events) = normalize_top_level_system_text(
                &input,
                ClaudeCodePromptNormalization::StablePrefixV1,
            );
            assert_eq!(normalized, input);
            assert_eq!(events.total(), 0);
        }
    }

    #[test]
    fn task_nudge_after_tools_heading_fails_open() {
        let input = format!("# Tools\nbody\n{TASK_NUDGE}");

        let (normalized, events) = normalize_top_level_system_text(
            &input,
            ClaudeCodePromptNormalization::StablePrefixV1,
        );

        assert_eq!(normalized, input);
        assert_eq!(events.total(), 0);
    }

    #[test]
    fn duplicate_task_or_date_candidates_fail_open() {
        let duplicate_task = format!("{TASK_NUDGE}\n{TASK_NUDGE}\n# Tools");
        let duplicate_date = format!(
            "{}\n{}\n# Tools",
            date_reminder("2026-07-19"),
            date_reminder("2026-07-20")
        );
        for input in [duplicate_task, duplicate_date] {
            let (normalized, events) = normalize_top_level_system_text(
                &input,
                ClaudeCodePromptNormalization::StablePrefixV1,
            );
            assert_eq!(normalized, input);
            assert_eq!(events.total(), 0);
        }
    }

    #[test]
    fn protected_system_signals_remain_byte_for_byte_unchanged() {
        let protected = "Context threshold alert: 12% remains.\nFile modification notice: src/main.rs changed.\n<system-reminder>hook feedback</system-reminder>\n<system-reminder>IMPORTANT: A new message arrived mid-turn.</system-reminder>";
        let (normalized, events) = normalize_top_level_system_text(
            protected,
            ClaudeCodePromptNormalization::StablePrefixV1,
        );
        assert_eq!(normalized.as_bytes(), protected.as_bytes());
        assert_eq!(events.total(), 0);
    }

    #[test]
    fn protected_state_adjacent_to_removable_blocks_is_byte_exact_with_crlf() {
        let before = "Context threshold alert: 12% remains.\tFile state: dirty.";
        let after = "<system-reminder>IMPORTANT: A new message arrived mid-turn.</system-reminder>";
        let input = format!(
            "{before}\r\n{TASK_NUDGE}\r\n\r\n{}\r\n\r\n# Tools\r\n{after}",
            date_reminder("2026-07-20")
        );
        let expected = format!("{before}\r\n# Tools\r\n{after}");

        let (normalized, events) =
            normalize_top_level_system_text(&input, ClaudeCodePromptNormalization::StablePrefixV1);

        assert_eq!(normalized.as_bytes(), expected.as_bytes());
        assert_eq!(events.task_nudges, 1);
        assert_eq!(events.date_reminders, 1);
    }

    #[test]
    fn cache_namespace_is_stable_within_policy_and_distinct_when_enabled() {
        let baseline = "template-digest";
        let off = ClaudeCodePromptNormalization::Off.namespace_template_sig(baseline);
        let first = ClaudeCodePromptNormalization::StablePrefixV1.namespace_template_sig(baseline);
        let second = ClaudeCodePromptNormalization::StablePrefixV1.namespace_template_sig(baseline);
        assert_eq!(off, baseline);
        assert_eq!(first, second);
        assert_ne!(off, first);
    }
}
