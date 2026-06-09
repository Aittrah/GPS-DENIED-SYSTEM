---
description: Scaffold a new VNS plugin skill interactively. Prompts for skill name, trigger, and steps, then creates skills/<name>/SKILL.md pre-populated with the standard VNS template (ROS 2 + python3 + pip3 prerequisites baked in).
---

# VNS: Create Skill

Scaffold a new skill for this plugin at:
  `/home/hp/GPS-DENIED-SYSTEM/.claude/plugin/skills/<name>/SKILL.md`

## Arguments

`$ARGUMENTS` — optional skill name in kebab-case (e.g. `check-battery`).

## Instructions

**Step 1 — Determine skill name**

If `$ARGUMENTS` is provided, use it as the skill name (convert to lowercase kebab-case, replace spaces with dashes).  Otherwise ask:

> "What should this skill be named? (kebab-case, e.g. `calibrate-camera`)"

**Step 2 — Gather information**

Ask the user these three questions (can be answered together or one at a time):

1. "Describe in one sentence when Claude should trigger this skill (the `description:` frontmatter value)."
2. "List the ordered steps for this skill — one step per line.  Be specific about commands, file paths, and expected outputs.  Press Enter twice when done."
3. "List any common errors or gotchas to document (optional — press Enter to skip)."

**Step 3 — Create the directory and file**

Create directory:
  `/home/hp/GPS-DENIED-SYSTEM/.claude/plugin/skills/<name>/`

Write the file:
  `/home/hp/GPS-DENIED-SYSTEM/.claude/plugin/skills/<name>/SKILL.md`

Use this exact template, substituting the user's answers:

```markdown
---
description: <trigger description from Step 2 question 1>
---

# <Title Case of skill name>

## Purpose
<One paragraph summarising what this skill accomplishes>

## Prerequisites

```bash
# Standard VNS environment setup (ALWAYS required)
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models
```

> Rules: use `python3` (not `python`), `pip3 install` (not `pip install`),
> absolute paths with `/home/hp/` base (not `~/`).

## Steps

<Numbered list of steps with bash code blocks from Step 2 question 2>

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
<Rows from Step 2 question 3, or "None documented yet." if skipped>

## Related Files

- <!-- list any source files relevant to this skill -->
```

**Step 4 — Confirm and instruct**

After writing the file:
1. Print the absolute path of the created file.
2. Show the full file content.
3. Remind the user: **Run `/reload-plugins` to make the new skill available as `/vns:<name>`.**
