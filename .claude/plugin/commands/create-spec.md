---
description: Create a structured specification file for a VNS functional requirement. Prompts for FR ID, title, requirement text, and metadata, then writes specs/<FR-ID>.md with full acceptance criteria, test plan, and linked source files.
---

# VNS: Create Spec

Create a structured specification file at:
  `/home/hp/GPS-DENIED-SYSTEM/specs/<FR-ID>.md`

## Arguments

`$ARGUMENTS` — optional FR ID (e.g. `FR-5`).

## Instructions

**Step 1 — Determine FR ID**

If `$ARGUMENTS` is provided, normalise it to uppercase with hyphen (e.g. `fr5` → `FR-5`).  Otherwise ask:

> "Enter the Functional Requirement ID (e.g. `FR-5`):"

**Step 2 — Gather information**

Ask the user:

1. "Title (short noun phrase, e.g. `UAV Image Preprocessing`):"
2. "Requirement text — verbatim from SRS, or describe what the system **shall** do:"
3. "Rationale — why is this requirement needed?"
4. "Acceptance criteria — list at least 3 testable, numbered criteria:"
5. "Dependencies — other FR IDs this depends on (comma-separated, or `None`):"
6. "Linked source files — comma-separated paths relative to project root:"

**Step 3 — Create specs directory (if needed)**

```
/home/hp/GPS-DENIED-SYSTEM/specs/
```

**Step 4 — Write the spec file**

Path: `/home/hp/GPS-DENIED-SYSTEM/specs/<FR-ID>.md`

Use this template:

```markdown
# <FR-ID>: <Title>

## Requirement
> <Verbatim requirement text>

## Rationale
<Why this requirement exists>

## Acceptance Criteria
1. <Testable criterion>
2. <Testable criterion>
3. <Testable criterion>

## Dependencies
<Other FR-IDs, or "None">

## Linked Source Files
- `<relative/path>` — <role this file plays>

## Test Plan

### Unit Tests
- <What unit-level tests should verify>

### Integration Tests
- <What integration-level tests should verify>

### Simulation Tests
- <What Gazebo simulation tests should verify>

## Status
- [ ] Not Started
- [ ] In Progress
- [ ] Done
```

**Step 5 — Confirm**

Print the absolute path and the file content.  Do NOT suggest running `/reload-plugins` (spec files are not plugin components).
