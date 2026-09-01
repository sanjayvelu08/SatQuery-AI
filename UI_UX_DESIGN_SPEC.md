# SatQuery AI — UI/UX Design Specification v1.0

> **Status:** Design spec only — no implementation yet.
> **Target stack:** React 18 + Vite + TypeScript + Tailwind CSS
> **Backend:** Existing Python SatQuery pipeline (frozen, unchanged)
> **Hardware context:** NVIDIA RTX 3050 (4 GB), ISRO SIH26167 prototype

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Design System — Tokens & Primitives](#2-design-system)
3. [Component Library](#3-component-library)
4. [Screen Specifications](#4-screen-specifications)
5. [Backend ↔ Frontend Data Mapping](#5-backend-data-mapping)
6. [Animation & Motion](#6-animation--motion)
7. [Responsive Behavior](#7-responsive-behavior)
8. [Accessibility & Performance](#8-accessibility--performance)

---

## 1. Design Philosophy

### Visual Identity

SatQuery AI is an **ISRO / Earth Observation / Satellite Intelligence** tool. The UI must feel like a product used by remote-sensing analysts — not a startup demo, not a military command center, and not a generic chatbot.

**Three words:** Professional · Precise · Scientific

### Aesthetic Direction

| Aspect | Direction |
|--------|-----------|
| **Overall** | Clean, information-dense, dark-background scientific tool |
| **Inspiration** | ISRO mission dashboards, ESA Earth Explorer UIs, QGIS dark theme |
| **Background** | Deep navy with subtle gradient — never pure black |
| **Accents** | Teal/cyan for primary actions, earth-green for secondary |
| **Typography** | Clean sans-serif (Inter for body, JetBrains Mono for data) |
| **Cards** | Rounded, semi-transparent glass-like panels with subtle borders |
| **Imagery** | Satellite images ARE the visual focus — never decorative |
| **Motion** | Smooth, restrained — 200-300ms ease transitions |
| **Density** | High — maximize information per viewport without clutter |

### Color Palette — ISRO/Scientific

```
Primary background:     #0A1628  (deep space navy)
Secondary background:   #0F2035  (slightly lighter navy)
Card background:        #132842  (panel navy, 85% opacity)
Card border:            #1E3A5F  (subtle steel blue)
Teal accent:            #00D4AA  (primary action — satellite teal)
Teal dark:              #00A882  (hover states)
Cyan glow:              #00E5FF  (highlights, active states)
Earth green:            #4CAF50  (success, vegetation, SAR)
Amber:                  #FFB74D  (warnings, classification)
Coral:                  #FF6B6B  (errors, critical alerts)
White primary:          #E8F4FD  (headings, primary text)
White secondary:        #94B3CC  (body text)
White muted:            #5A7A96  (labels, secondary info)
Grid lines:             rgba(0, 212, 170, 0.06)  (subtle background grid)
```

---

## 2. Design System — Tokens & Primitives

### 2.1 Color Tokens

```
--color-bg-primary:        #0A1628
--color-bg-secondary:      #0F2035
--color-bg-card:           rgba(19, 40, 66, 0.85)
--color-bg-card-solid:     #132842
--color-bg-card-hover:     #1A3352
--color-bg-input:          #0D1E33
--color-bg-input-focus:    #102540

--color-border:            #1E3A5F
--color-border-light:      #2A4A6B
--color-border-focus:      #00D4AA

--color-accent-teal:       #00D4AA
--color-accent-teal-hover: #00A882
--color-accent-cyan:       #00E5FF
--color-accent-green:      #4CAF50
--color-accent-amber:      #FFB74D
--color-accent-coral:      #FF6B6B

--color-text-primary:      #E8F4FD
--color-text-secondary:    #94B3CC
--color-text-muted:        #5A7A96
--color-text-inverse:      #0A1628

--color-intent-caption:    #4FC3F7
--color-intent-vqa:        #AB47BC
--color-intent-detect:     #EF5350
--color-intent-grounding:  #FF7043
--color-intent-classify:   #FFA726
--color-intent-sar:        #26A69A
--color-intent-general:    #78909C

--color-status-success:    #4CAF50
--color-status-loading:    #00D4AA
--color-status-error:      #FF6B6B
--color-status-unsupported:#FFB74D
```

### 2.2 Typography Scale

```
Font family (body):      'Inter', -apple-system, system-ui, sans-serif
Font family (mono):      'JetBrains Mono', 'Fira Code', monospace

--text-hero:    2rem / 700 / -0.02em tracking   (page titles)
--text-h1:      1.5rem / 600 / -0.01em           (section headers)
--text-h2:      1.125rem / 600 / 0               (card titles)
--text-h3:      1rem / 600 / 0.02em tracking     (subsection)
--text-body:    0.9375rem / 400 / 0.01em         (main content)
--text-small:   0.8125rem / 400 / 0.02em         (labels, metadata)
--text-tiny:    0.75rem / 500 / 0.04em tracking  (badges, chips)
--text-mono:    0.8125rem / 400 / 0              (data, timing, code)
```

### 2.3 Spacing Scale

```
--space-xs:     4px
--space-sm:     8px
--space-md:     12px
--space-lg:     16px
--space-xl:     24px
--space-2xl:    32px
--space-3xl:    48px
```

### 2.4 Border Radius

```
--radius-sm:    6px     (chips, badges, small buttons)
--radius-md:    8px     (inputs, cards)
--radius-lg:    12px    (modals, panels)
--radius-xl:    16px    (hero cards)
--radius-full:  9999px  (circular avatars, pills)
```

### 2.5 Shadows

```
--shadow-sm:    0 1px 3px rgba(0,0,0,0.3)
--shadow-md:    0 4px 12px rgba(0,0,0,0.4)
--shadow-lg:    0 8px 24px rgba(0,0,0,0.5)
--shadow-glow:  0 0 20px rgba(0,212,170,0.15)    (teal glow for focus)
```

---

## 3. Component Library

### 3.1 Buttons

#### Primary Button
```css
/* "Analyze" action */
background: linear-gradient(135deg, #00D4AA, #00A882)
color: #0A1628 (dark text on teal)
font: 600 0.875rem Inter
padding: 10px 24px
border-radius: 6px
box-shadow: 0 2px 8px rgba(0,212,170,0.3)
transition: all 200ms ease
/* hover */  transform: translateY(-1px); box-shadow glow increases
/* active */ transform: translateY(0); opacity: 0.9
/* disabled */ opacity: 0.4; cursor: not-allowed
```

#### Secondary Button
```css
/* "Clear", "Cancel" actions */
background: transparent
border: 1px solid var(--color-border-light)
color: var(--color-text-secondary)
font: 500 0.875rem Inter
padding: 10px 24px
border-radius: 6px
/* hover */  border-color: var(--color-text-muted); color: var(--color-text-primary)
```

#### Ghost Button
```css
/* Toolbar actions, small inline actions */
background: transparent
color: var(--color-accent-teal)
font: 500 0.8125rem Inter
padding: 6px 12px
border-radius: 4px
/* hover */  background: rgba(0,212,170,0.08)
```

### 3.2 Cards

#### Standard Card
```css
background: var(--color-bg-card)
backdrop-filter: blur(12px)
border: 1px solid var(--color-border)
border-radius: var(--radius-lg)
padding: var(--space-xl)
box-shadow: var(--shadow-sm)
transition: border-color 200ms ease
/* hover */  border-color: var(--color-border-light)
```

#### Feature Card (Analysis Result)
```css
/* Same as standard + accent top border */
border-top: 3px solid var(--color-accent-teal)
```

#### Input Card
```css
/* Image upload area, query input area */
background: var(--color-bg-card)
border: 1px solid var(--color-border)
border-radius: var(--radius-lg)
padding: var(--space-xl)
/* focus-within */  border-color: var(--color-accent-teal); box-shadow: var(--shadow-glow)
```

### 3.3 Badges

#### Intent Badge
```css
display: inline-flex
align-items: center
gap: 4px
padding: 3px 10px
border-radius: var(--radius-full)
font: 500 0.6875rem Inter (uppercase, 0.06em tracking)
/* Each intent gets its own color */
/* e.g., caption → blue-500 bg with 15% opacity, blue-300 text */
```

#### Status Badge
```css
/* Success / Loading / Error / Unsupported */
display: inline-flex
align-items: center
gap: 4px
padding: 2px 8px
border-radius: var(--radius-full)
font: 500 0.6875rem Inter
```

### 3.4 Inputs

#### Text Input
```css
background: var(--color-bg-input)
border: 1px solid var(--color-border)
border-radius: var(--radius-md)
padding: 10px 14px
color: var(--color-text-primary)
font: 400 0.9375rem Inter
/* placeholder */ color: var(--color-text-muted)
/* focus */ border-color: var(--color-accent-teal); box-shadow: var(--shadow-glow)
```

#### Image Upload Area
```css
background: var(--color-bg-input)
border: 2px dashed var(--color-border)
border-radius: var(--radius-lg)
min-height: 280px
/* When hovering with file */ border-color: var(--color-accent-teal)
/* When image loaded */ border-style: solid; border-color: var(--color-border)
```

### 3.5 Dropdown / Select
```css
background: var(--color-bg-card-solid)
border: 1px solid var(--color-border)
border-radius: var(--radius-md)
padding: 8px 12px
color: var(--color-text-primary)
font: 400 0.875rem Inter
/* dropdown options */ background: var(--color-bg-secondary)
```

### 3.6 Detection Table (SAR + Grounding)
```css
/* Used for structured detection results */
background: rgba(0,0,0,0.2)
border-radius: var(--radius-md)
border-collapse: separate
border-spacing: 0

th:
  background: rgba(0,212,170,0.08)
  color: var(--color-text-secondary)
  font: 600 0.75rem Inter (uppercase, 0.05em tracking)
  padding: 8px 12px
  border-bottom: 1px solid var(--color-border)

td:
  color: var(--color-text-primary)
  font: 400 0.8125rem Inter
  padding: 8px 12px
  border-bottom: 1px solid rgba(30,58,95,0.5)

tr:hover td:
  background: rgba(0,212,170,0.04)
```

### 3.7 Confidence Bar
```css
/* Horizontal bar showing detection confidence */
height: 6px
border-radius: 3px
background: rgba(255,255,255,0.1)
/* fill */ background: linear-gradient(90deg, var(--color-accent-teal), var(--color-accent-green))
/* confidence < 30% */ background: var(--color-accent-coral)
/* confidence 30-70% */ background: var(--color-accent-amber)
/* confidence > 70% */ background: var(--color-accent-green)
```

### 3.8 Loading States

#### Spinner (global)
```css
/* Teal ring spinner */
width: 24px; height: 24px
border: 2px solid var(--color-border)
border-top-color: var(--color-accent-teal)
border-radius: 50%
animation: spin 800ms linear infinite
```

#### Analysis Loading Overlay
```css
/* Full-panel loading state during VLM inference (50-260s) */
background: var(--color-bg-card)
border-radius: var(--radius-lg)
padding: var(--space-3xl)
text-align: center

Elements (top to bottom):
  1. Orbital animation (SVG path drawing)
  2. "Analyzing satellite imagery..." (text-body, muted)
  3. Progress stages indicator (see below)
  4. Elapsed time counter (mono font, updating every 1s)
  5. "EarthDial 4B loading" / "Running inference" status
```

#### Progress Stages (Analysis Loading)
```
○ Route query        →   ● Analyzing imagery    →   ○ Generating response
         (instant)              (50-260s)                   (1-3s)

Stage indicators: circle (pending), filled circle with glow (active), checkmark (done)
Connected by horizontal line, teal when active, muted when pending
```

### 3.9 Tooltip
```css
background: var(--color-bg-card-solid)
border: 1px solid var(--color-border)
border-radius: var(--radius-sm)
padding: 6px 10px
font: 400 0.75rem Inter
color: var(--color-text-secondary)
box-shadow: var(--shadow-md)
max-width: 240px
```

---

## 4. Screen Specifications

---

### 4.1 Screen: Landing / Home

**Purpose:** First impression. Show what SatQuery AI does before any image is uploaded.

#### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER BAR (fixed, compact)                                    │
│  [🛰️ SatQuery AI]                    [About]  [GitHub]         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    (centered, 60% max-width)                    │
│                                                                 │
│              🛰️  SATELLITE INTELLIGENCE                         │
│              AT YOUR COMMAND                                    │
│                                                                 │
│     Upload a satellite image. Ask a natural-language question.  │
│     Get AI-powered analysis in seconds.                         │
│                                                                 │
│         ┌──────────────────────────────────────┐                │
│         │  [📁 Upload Satellite Image]         │                │
│         │  or drag and drop                    │                │
│         └──────────────────────────────────────┘                │
│                                                                 │
│  ─── CAPABILITIES ────────────────────────────────────────      │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ 📝       │  │ ❓       │  │ 🎯       │  │ 📡       │       │
│  │ Caption  │  │ VQA      │  │ Detect   │  │ SAR      │       │
│  │ Describe │  │ Ask      │  │ Locate   │  │ Ships    │       │
│  │ imagery  │  │ questions│  │ features │  │ in radar │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
│  ─── HOW IT WORKS ──────────────────────────────────────────    │
│                                                                 │
│  [Upload]  →  [AI Routes]  →  [Inference]  →  [Results]       │
│    1              2               3               4             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Components
| Component | Description |
|-----------|-------------|
| **Header** | Compact bar: logo + app name (left), nav links (right) |
| **Hero** | Large title + subtitle + primary CTA button |
| **Capability Cards** | 4 cards in a row: Caption, VQA, Detect, SAR |
| **How It Works** | 4-step horizontal flow with icons and labels |
| **Background** | Deep navy with subtle animated grid lines |

#### Colors & Typography
- Hero title: `--text-hero`, white, letter-spacing: -0.02em
- Hero subtitle: `--text-body`, `--color-text-secondary`
- Capability cards: glass-card style, icon above text
- Background gradient: linear from `#0A1628` to `#0F2035`

#### Animations
- Hero title: fade-in + slight upward translate (0.6s ease)
- Capability cards: staggered fade-in (0.1s delay each)
- Background grid: very slow subtle drift (CSS animation, 60s loop)
- CTA button: subtle pulse glow on idle (2s loop, teal)

#### Responsive
- **≥1200px:** Full layout as designed
- **768-1199px:** Capability cards → 2×2 grid
- **<768px:** Single column, capability cards stacked, hero smaller

---

### 4.2 Screen: Analysis Workspace — Empty

**Purpose:** Image uploaded but no query yet. Ready for analysis.

#### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER BAR                                                     │
│  [🛰️ SatQuery AI]    [Demo: Dropdown ▾]   [About]  [GitHub]  │
├────────────────────────────────┬────────────────────────────────┤
│                                │                                │
│  SATELLITE IMAGE               │  ANALYSIS RESULT               │
│  ┌──────────────────────────┐  │  ┌──────────────────────────┐  │
│  │                          │  │  │                          │  │
│  │     [Uploaded Image]     │  │  │   No analysis yet.       │  │
│  │                          │  │  │                          │  │
│  │                          │  │  │   Upload an image and    │  │
│  │     (with zoom/pan)      │  │  │   ask a question, or    │  │
│  │                          │  │  │   select a demo above.  │  │
│  └──────────────────────────┘  │  │                          │  │
│                                │  │   Supported queries:     │  │
│  QUERY                         │  │   • Describe this image  │  │
│  ┌──────────────────────────┐  │  │   • Are there buildings? │  │
│  │ Describe this image...   │  │  │   • Detect ships (SAR)   │  │
│  └──────────────────────────┘  │  └──────────────────────────┘  │
│                                │                                │
│  [🔍 Analyze]    [Clear]       │  VISUAL EVIDENCE               │
│                                │  ┌──────────────────────────┐  │
│                                │  │                          │  │
│                                │  │   Annotated image will   │  │
│                                │  │   appear here after      │  │
│                                │  │   detection/grounding.   │  │
│                                │  │                          │  │
│                                │  └──────────────────────────┘  │
│                                │                                │
├────────────────────────────────┴────────────────────────────────┤
│  📜 QUERY HISTORY: (empty state)                                │
└─────────────────────────────────────────────────────────────────┘
```

#### Components
| Component | State |
|-----------|-------|
| **Image panel** | Shows uploaded image with zoom/pan controls |
| **Query input** | Text input with placeholder suggestions |
| **Analyze button** | Primary, enabled when image + query present |
| **Clear button** | Secondary, resets image + query |
| **Result panel** | Empty state with supported query types |
| **Visual Evidence** | Placeholder with dashed border, icon |
| **History** | "No queries yet" muted text |

#### Colors
- Image panel: card with 3px teal top border
- Empty state text: `--color-text-muted`
- Placeholder icons: `--color-text-muted` at 30% opacity

---

### 4.3 Screen: Analysis Workspace — Image Uploaded

**Purpose:** Image is showing. Query area is prominent.

#### Changes from Empty
- Image displays in panel with filename + dimensions shown below
- Query input is focused (glowing border)
- Query suggestions appear as ghost-button chips below input:
  ```
  [Describe this] [Are there buildings?] [Classify land use] [Detect features]
  ```
- Analyze button is fully active with teal gradient

#### Suggestion Chips
```css
background: rgba(0,212,170,0.08)
border: 1px solid rgba(0,212,170,0.2)
color: var(--color-accent-teal)
font: 500 0.75rem Inter
padding: 4px 10px
border-radius: var(--radius-full)
/* hover */  background: rgba(0,212,170,0.15); border-color: var(--color-accent-teal)
```

---

### 4.4 Screen: Analysis — Loading

**Purpose:** Inference in progress. Critical for user confidence during 50-260s wait.

#### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  (same header + image panel as before)                          │
├────────────────────────────────┬────────────────────────────────┤
│  SATELLITE IMAGE               │  ANALYZING...                  │
│  ┌──────────────────────────┐  │  ┌──────────────────────────┐  │
│  │                          │  │  │                          │  │
│  │     [Original Image]     │  │  │    🛰️ (orbital SVG)     │  │
│  │     (slightly dimmed)    │  │  │                          │  │
│  │                          │  │  │  Analyzing satellite     │  │
│  │                          │  │  │  imagery...              │  │
│  └──────────────────────────┘  │  │                          │  │
│                                │  │  ● Route query ✓         │  │
│  QUERY                         │  │  ● Analyzing imagery ◉   │  │
│  ┌──────────────────────────┐  │  │  ○ Generating response   │  │
│  │ Detect ships in SAR      │  │  │                          │  │
│  └──────────────────────────┘  │  │  ⏱️ 00:12 elapsed        │  │
│                                │  │                          │  │
│  [⏳ Analyzing...]  [Cancel]   │  │  Model: EarthDial 4B RGB │  │
│                                │  └──────────────────────────┘  │
│                                │                                │
├────────────────────────────────┴────────────────────────────────┤
│  📜 QUERY HISTORY: (previous queries still visible)             │
└─────────────────────────────────────────────────────────────────┘
```

#### Components
| Component | State |
|-----------|-------|
| **Analyze button** | Changed to "⏳ Analyzing..." with spinner, disabled |
| **Cancel button** | Appears (secondary), enables abort |
| **Result panel** | Replaced by loading overlay |
| **Image** | Slightly dimmed (opacity: 0.85) |
| **Progress indicator** | 3-stage horizontal with animated active dot |
| **Elapsed timer** | Updates every second, mono font |
| **Model info** | Shows which model is running |

#### Animations
- Orbital SVG: continuous slow rotation (4s loop)
- Active stage dot: pulsing glow (1.5s loop)
- Progress line: animated gradient when active stage is in progress
- Elapsed counter: smooth digit transition

#### Transition to Result
When analysis completes:
- Loading overlay fades out (300ms)
- Result content fades in (400ms, slight upward translate)
- Visual evidence slides in from right (300ms)

---

### 4.5 Screen: Optical Analysis Result

**Purpose:** EarthDial VLM returned a caption/VQA/classification/grounding result.

#### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER BAR                                                     │
├────────────────────────────────┬────────────────────────────────┤
│  SATELLITE IMAGE               │  ANALYSIS RESULT               │
│  ┌──────────────────────────┐  │  ┌──────────────────────────┐  │
│  │                          │  │  │ 📝 Captioning     ✅     │  │
│  │     [Original Image]     │  │  ├──────────────────────────┤  │
│  │                          │  │  │                          │  │
│  └──────────────────────────┘  │  │  This Sentinel-2 image   │  │
│                                │  │  captures an agricultural│  │
│  QUERY                         │  │  region with active      │  │
│  ┌──────────────────────────┐  │  │  cropland and fallow     │  │
│  │ Describe this image      │  │  │  fields...               │  │
│  └──────────────────────────┘  │  │                          │  │
│                                │  │  [full markdown answer]  │  │
│  [🔍 Analyze]    [Clear]       │  │                          │  │
│                                │  │                          │  │
│                                │  ├──────────────────────────┤  │
│                                │  │ 🛰️ EarthDial 4B RGB      │  │
│                                │  │ ⏱️ Route 2ms · VLM 52s   │  │
│                                │  └──────────────────────────┘  │
│                                │                                │
│                                │  VISUAL EVIDENCE               │
│                                │  ┌──────────────────────────┐  │
│                                │  │                          │  │
│                                │  │  [Annotated image with   │  │
│                                │  │   bounding boxes if      │  │
│                                │  │   detection/grounding]   │  │
│                                │  │                          │  │
│                                │  │  — OR —                  │  │
│                                │  │                          │  │
│                                │  │  (empty state: "Bounding │  │
│                                │  │   boxes available for    │  │
│                                │  │   detection queries")    │  │
│                                │  │                          │  │
│                                │  └──────────────────────────┘  │
│                                │                                │
├────────────────────────────────┴────────────────────────────────┤
│  📜 QUERY HISTORY:                                               │
│  📝 Caption > "Describe this image"                              │
└─────────────────────────────────────────────────────────────────┘
```

#### Components
| Component | Description |
|-----------|-------------|
| **Result header** | Intent badge + status indicator (right-aligned) |
| **Answer** | Rendered Markdown with tables, lists, bold, italic |
| **Footer strip** | Model name + timing in monospace |
| **Visual Evidence** | Annotated image (if grounding/detect) or empty state |
| **History** | New entry prepended |

#### Result Panel Details
- Answer rendered as Markdown (using react-markdown with remark-gfm for tables)
- Scrollable if answer is long (max-height with custom scrollbar)
- Custom scrollbar: thin, teal track, navy thumb

#### Visual Evidence States
| Condition | Display |
|-----------|---------|
| Intent = detect/grounding + has annotated image | Annotated image with bbox overlay |
| Intent = caption/vqa/classification | Empty state: "Bounding boxes available for detection queries" |
| Intent = sar | SAR annotated image (see SAR screen) |
| Annotation failed | Muted message: "Could not generate visual evidence" |

---

### 4.6 Screen: SAR Vessel Detection Result

**Purpose:** YOLOv8 SAR detector returned ship detection results.

#### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER BAR                                                     │
├────────────────────────────────┬────────────────────────────────┤
│  SAR IMAGE                     │  ANALYSIS RESULT               │
│  ┌──────────────────────────┐  │  ┌──────────────────────────┐  │
│  │                          │  │  │ 📡 SAR Analysis    ✅    │  │
│  │  [SAR image with         │  │  ├──────────────────────────┤  │
│  │   bounding boxes drawn]  │  │  │                          │  │
│  │                          │  │  │  Detected 3 maritime     │  │
│  │  ┌──────┐                │  │  │  targets in SAR imagery. │  │
│  │  │Ship  │                │  │  │                          │  │
│  │  │76.3% │                │  │  │  # │ Object │ Conf. │   │  │
│  │  └──────┘                │  │  │  1 │ Ship   │ 76.3% │   │  │
│  │         ┌────┐           │  │  │  2 │ Ship   │ 26.8% │   │  │
│  │         │Ship│           │  │  │  3 │ Ship   │ 25.3% │   │  │
│  │         └────┘           │  │  │                          │  │
│  └──────────────────────────┘  │  │  Confidence distribution:│  │
│                                │  │  ████░░░░░░ 76.3%        │  │
│  QUERY                         │  │  █░░░░░░░░░ 26.8%        │  │
│  ┌──────────────────────────┐  │  │  █░░░░░░░░░ 25.3%        │  │
│  │ Detect ships in SAR      │  │  │                          │  │
│  └──────────────────────────┘  │  ├──────────────────────────┤  │
│                                │  │ 🛰️ YOLOv8 SAR Detector   │  │
│  [🔍 Analyze]    [Clear]       │  │ ⏱️ Route 1ms · Det 3.2s  │  │
│                                │  │ 💾 21 MB VRAM            │  │
│                                │  └──────────────────────────┘  │
│                                │                                │
│                                │  VISUAL EVIDENCE               │
│                                │  ┌──────────────────────────┐  │
│                                │  │  [Annotated SAR image    │  │
│                                │  │   with ship bounding     │  │
│                                │  │   boxes and labels]      │  │
│                                │  └──────────────────────────┘  │
├────────────────────────────────┴────────────────────────────────┤
│  📜 QUERY HISTORY:                                               │
│  📡 SAR > "Detect ships in this SAR image"                      │
└─────────────────────────────────────────────────────────────────┘
```

#### SAR-Specific Components

**Detection Table:**
```
| # | Object | Confidence | Bounding Box    | Bar          |
|---|--------|-----------|-----------------|--------------|
| 1 | Ship   | 76.3%     | [240,243,285,297] | ████████░░ |
| 2 | Ship   | 26.8%     | [144,1,188,26]    | ███░░░░░░░ |
| 3 | Ship   | 25.3%     | [243,244,296,307] | ███░░░░░░░ |
```

**Confidence Bar Colors:**
- >70%: green gradient (high confidence)
- 30-70%: amber gradient (medium)
- <30%: coral gradient (low)

**SAR Footer:**
- Shows VRAM usage (unique to SAR, not shown for optical)
- Shows inference time in ms (not seconds, since it's fast)

#### Annotated Image
- SAR annotated image displayed in the Visual Evidence panel
- Drawn by existing `visualize.py` with bounding boxes + labels + confidence
- Image shows original SAR with color-coded boxes overlaid

---

### 4.7 Screen: Analysis History

**Purpose:** View past queries in this session. Appears as a collapsible panel below the main workspace.

#### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  📜 QUERY HISTORY                              [Clear History]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ 1 ──────────────────────────────────────────────────────┐  │
│  │ 📡 SAR Detection │ "Detect ships in SAR"                 │  │
│  │ YOLOv8 SAR • 3 targets • 3.2s • ✅                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ 2 ──────────────────────────────────────────────────────┐  │
│  │ 📝 Captioning │ "Describe this satellite image"          │  │
│  │ EarthDial 4B • 52.1s • ✅                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ 3 ──────────────────────────────────────────────────────┐  │
│  │ 🎯 Detection │ "Locate main features"                    │  │
│  │ EarthDial 4B • 262.3s • ✅ • 📍 3 regions                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Components
| Component | Description |
|-----------|-------------|
| **History header** | Count badge + title + Clear button |
| **History entry** | Numbered card with intent badge, query text, model, timing, status |
| **Click to re-view** | Clicking an entry loads its result back into the result panel |

#### History Entry Card
```css
background: var(--color-bg-card)
border: 1px solid var(--color-border)
border-left: 3px solid var(--color-intent-{type})
border-radius: var(--radius-md)
padding: var(--space-md) var(--space-lg)

Layout:
  Row 1: [Number] [Intent badge] [Query text — truncated with ellipsis]
  Row 2: [Model name] • [Timing] • [Status] • [Extra info if applicable]
```

#### Responsive
- **≥1200px:** Full-width row entries
- **<768px:** Entries stack, query text truncated earlier

---

### 4.8 Screen: About / System Architecture

**Purpose:** Modal overlay explaining the system to evaluators. Accessible from header "About" link.

#### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│                                                 [✕ Close]       │
│                                                                 │
│  🛰️ SATQUERY AI — SYSTEM OVERVIEW                              │
│  Remote Sensing Vision-Language Assistant                       │
│  ISRO SIH 2026 · Problem Statement SIH26167                    │
│                                                                 │
│  ─── ARCHITECTURE ──────────────────────────────────────────    │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ 📷 Image │ →  │ 🧠 Router │ →  │ 🤖 Model │                  │
│  │  Upload  │    │ Keywords │    │ Inference│                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│       │                               │                         │
│       │              ┌────────────────┤                         │
│       │              │                │                         │
│       ▼              ▼                ▼                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐                  │
│  │ 🛰️ SAR  │  │ 🌍 VLM   │  │ 📍 Visual    │                  │
│  │ YOLOv8   │  │EarthDial │  │  Grounding   │                  │
│  └──────────┘  └──────────┘  └──────────────┘                  │
│       │              │                │                         │
│       └──────────────┴────────────────┘                         │
│                      │                                          │
│                      ▼                                          │
│              ┌──────────────┐                                   │
│              │ 📊 Results   │                                   │
│              │ + Visual     │                                   │
│              │   Evidence   │                                   │
│              └──────────────┘                                   │
│                                                                 │
│  ─── MODELS ────────────────────────────────────────────────    │
│                                                                 │
│  | Model | Purpose | Size | VRAM | Speed |                     │
│  |-------|---------|------|------|-------|                     │
│  | EarthDial 4B | Optical VLM | 8 GB | 2.85 GB | 50-260s |   │
│  | YOLOv8n | SAR vessel det. | 6 MB | 21 MB | 3-7ms |         │
│  | Router | Intent classifier | 0 KB | 0 MB | <1ms |           │
│                                                                 │
│  ─── CAPABILITIES ──────────────────────────────────────────    │
│                                                                 │
│  ✅ Optical image captioning (Sentinel-2, Landsat, etc.)       │
│  ✅ Visual question answering on satellite imagery              │
│  ✅ Feature detection with bounding-box grounding               │
│  ✅ Scene classification (agricultural, urban, etc.)            │
│  ✅ SAR maritime vessel detection (YOLOv8)                      │
│  ⬜ Change detection (planned)                                  │
│  ⬜ Multi-image analysis (planned)                              │
│                                                                 │
│  ─── TECHNICAL DETAILS ─────────────────────────────────────    │
│                                                                 │
│  Hardware: NVIDIA RTX 3050 (4 GB)                               │
│  Backend: Python 3.12, PyTorch 2.5.1+cu121                     │
│  Frontend: React 18, Vite, TypeScript, Tailwind CSS            │
│  Inference: 4-bit quantized VLM via HuggingFace Transformers  │
│                                                                 │
│  ─── TEAM & ACKNOWLEDGMENTS ────────────────────────────────    │
│                                                                 │
│  Built for SIH 2026 · [GitHub link]                             │
│  Uses: EarthDial (InternVL + Phi-3), Ultralytics YOLOv8       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Modal Behavior
- Full overlay with backdrop blur (8px)
- Max-width: 800px, centered
- Scrollable if content overflows
- Close on X button or Escape key
- Entrance: fade-in (200ms) + scale from 0.95 to 1 (200ms)
- Exit: fade-out (150ms)

---

## 5. Backend ↔ Frontend Data Mapping

### 5.1 API Contract

The React frontend communicates with the Python backend via a REST API. We need a lightweight FastAPI wrapper around the existing `SatQueryPipeline`.

```typescript
// ── Request ──────────────────────────────────────────────────
interface AnalyzeRequest {
  image: File;              // Uploaded satellite image
  query: string;            // Natural-language query
  demo?: string;            // Optional: demo scenario name
}

// ── Response ─────────────────────────────────────────────────
interface AnalyzeResponse {
  // Core results
  query: string;
  intent: IntentType;
  all_intents: IntentType[];
  supported: boolean;
  answer: string | null;                // Markdown-formatted
  unsupported_reason: string;
  model_used: string;

  // Timing
  elapsed_route_ms: number;
  elapsed_vlm_s: number;
  elapsed_total_s: number;

  // Visual evidence
  annotated_image_url: string | null;   // URL to annotated image

  // SAR-specific
  sar_result: SARResult | null;
}

type IntentType =
  | 'caption'
  | 'vqa'
  | 'detect'
  | 'grounding'
  | 'classification'
  | 'change'
  | 'sar'
  | 'general';

interface SARResult {
  success: boolean;
  detections: SARDetection[];
  num_detections: number;
  inference_time_ms: number;
  gpu_vram_mb: number;
  error: string | null;
}

interface SARDetection {
  class_name: string;
  confidence: number;
  bbox_xyxy: number[];
}

// ── Demo Scenarios ──────────────────────────────────────────
interface DemoScenario {
  name: string;
  image_url: string;
  query: string;
  intent: IntentType;
  model_used: string;
  answer: string;           // Pre-computed Markdown
}
```

### 5.2 Backend Wrapper (FastAPI)

```python
# satquery/api.py — new file, wraps existing pipeline

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, sys, base64, json

app = FastAPI(title="SatQuery AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Mount annotated images directory
app.mount("/annotated", StaticFiles(directory="annotated"), name="annotated")

pipeline = SatQueryPipeline(max_history=10)

@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    query: str = Form(...),
    demo: str | None = Form(None),
):
    # Save uploaded image to temp file
    # Run pipeline
    # Return JSON matching AnalyzeResponse
    ...

@app.get("/api/demos")
async def get_demos():
    # Return list of demo scenarios with image URLs
    ...

@app.get("/api/health")
async def health():
    return {"status": "ok", "vram": SatQueryVLM.vram_info()}
```

### 5.3 Frontend Data Flow

```
User uploads image
     │
     ▼
React: FormData(image, query) → POST /api/analyze
     │
     ▼
FastAPI: save temp file → pipeline.run(path, query)
     │
     ├─→ RouteResult (intent classification, <1ms)
     │
     ├─→ If SAR: run_sar_detection() → SARResult
     │         → create_annotated_image() → annotated file
     │         → Return JSON with annotated_image_url
     │
     ├─→ If VLM: vlm.query() → InferenceResult (50-260s)
     │         → create_annotated_image() if grounding/detect
     │         → Return JSON with answer + annotated_image_url
     │
     ├─→ If unsupported: Return unsupported_reason
     │
     ▼
React: Receive JSON → Update state → Render appropriate screen
     │
     ├─→ intent="sar" → Render SAR Detection Result screen
     ├─→ intent="detect"/"grounding" → Render Optical Result + annotated image
     ├─→ intent="caption"/"vqa"/"classification" → Render Optical Result
     └─→ supported=false → Render Unsupported state
```

### 5.4 Demo Data Flow

```
User selects demo from dropdown
     │
     ▼
React: GET /api/demos → Receive list with pre-computed answers
     │
     ▼
React: Load demo image + pre-computed answer into state
     │
     ├─→ Instant render (no API call for answer)
     ├─→ Still call /api/annotate to generate visual evidence
     └─→ Answer is truthful pre-computed Markdown
```

### 5.5 State Management

```typescript
// Minimal state — no Redux needed
interface AppState {
  // Current session
  image: File | null;
  imagePreview: string;          // Object URL for display
  query: string;
  isAnalyzing: boolean;
  selectedDemo: string;

  // Current result
  result: AnalyzeResponse | null;
  annotatedImageUrl: string | null;

  // History (client-side, max 5)
  history: HistoryEntry[];
}

interface HistoryEntry {
  id: string;
  timestamp: number;
  query: string;
  intent: IntentType;
  model_used: string;
  elapsed_s: number;
  supported: boolean;
  result: AnalyzeResponse;
  imagePreview: string;
}
```

---

## 6. Animation & Motion

### 6.1 Timing Functions

```
--ease-standard:  cubic-bezier(0.4, 0, 0.2, 1)     (general transitions)
--ease-enter:     cubic-bezier(0, 0, 0.2, 1)         (elements entering)
--ease-exit:      cubic-bezier(0.4, 0, 1, 1)         (elements leaving)
--ease-bounce:    cubic-bezier(0.34, 1.56, 0.64, 1)  (subtle emphasis)
```

### 6.2 Animation Catalog

| Animation | Duration | Trigger | Description |
|-----------|----------|---------|-------------|
| **Page enter** | 400ms | Route change | Fade + translateY(-8px → 0) |
| **Card enter** | 300ms | New element | Fade + scale(0.97 → 1) |
| **Analysis loading** | Continuous | Analysis start | Orbital SVG rotation (4s) |
| **Progress pulse** | 1.5s loop | Active stage | Glow pulse on active dot |
| **Result appear** | 400ms | Analysis complete | Fade + translateY(8px → 0) |
| **Annotated image** | 300ms | Image ready | Scale(0.95 → 1) + fade |
| **Intent badge** | 200ms | State change | Background color cross-fade |
| **Hover lift** | 200ms | Mouse hover | translateY(-1px) + shadow increase |
| **Button press** | 100ms | Click | Scale(0.98 → 1) |
| **Tooltip appear** | 150ms | Hover | Fade + translateY(4px → 0) |
| **Modal open** | 200ms | Click | Fade + scale(0.95 → 1) |
| **Modal close** | 150ms | Click/Esc | Fade + scale(1 → 0.95) |
| **History slide** | 250ms | New entry | Slide down + fade |
| **Background grid** | 60s loop | Always | Slow translation drift |

### 6.3 Keyframe Definitions

```css
/* Orbital rotation for loading */
@keyframes orbital-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Pulse glow for active stage */
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,170,0.4); }
  50% { box-shadow: 0 0 12px 4px rgba(0,212,170,0.2); }
}

/* Subtle float for hero elements */
@keyframes subtle-float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-3px); }
}

/* Background grid drift */
@keyframes grid-drift {
  0% { background-position: 0 0; }
  100% { background-position: 40px 40px; }
}

/* Fade in up */
@keyframes fade-in-up {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Scale in */
@keyframes scale-in {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}
```

---

## 7. Responsive Behavior

### Breakpoints

```
--bp-sm:   640px    (mobile landscape)
--bp-md:   768px    (tablet)
--bp-lg:   1024px   (laptop)
--bp-xl:   1280px   (desktop)
--bp-2xl:  1536px   (large desktop)
```

### Layout Behavior

| Screen | ≥1280px | 1024-1279px | 768-1023px | <768px |
|--------|---------|-------------|------------|--------|
| **Main workspace** | 2 columns (50/50) | 2 columns (45/55) | Single column | Single column |
| **Landing capabilities** | 4 in a row | 4 in a row | 2×2 grid | Stacked |
| **History entries** | Full width | Full width | Full width | Truncated |
| **Header** | Full with all elements | Compact | Logo + hamburger | Logo only |
| **About modal** | 800px centered | 800px centered | 95% width | 95% width |
| **Visual Evidence** | Below result panel | Below result | Below result | Below result |
| **Detection table** | All columns | All columns | Object + Conf only | Object + Conf |

### Image Behavior

```
Desktop:  Image displayed at natural fit within panel (max 100% width)
Tablet:   Image displayed at natural fit (full width of column)
Mobile:   Image displayed at full width, max-height: 300px, object-fit: contain

All sizes: Click to open lightbox with zoom/pan controls
```

---

## 8. Accessibility & Performance

### 8.1 Accessibility

| Requirement | Implementation |
|-------------|----------------|
| **Keyboard nav** | Tab order: Header → Upload → Query → Analyze → Results |
| **Focus indicators** | Visible teal outline on all interactive elements |
| **Screen reader** | ARIA labels on all buttons, images, status indicators |
| **Color contrast** | All text meets WCAG AA (4.5:1 minimum on dark bg) |
| **Reduced motion** | Respect `prefers-reduced-motion: reduce` |
| **Alt text** | Uploaded images get descriptive alt from filename + query |
| **Error messages** | Live regions for analysis errors and status changes |

### 8.2 Performance Targets

| Metric | Target |
|--------|--------|
| **First Contentful Paint** | < 1s |
| **Largest Contentful Paint** | < 2s |
| **Time to Interactive** | < 2s |
| **Bundle size (gzipped)** | < 200 KB |
| **Analysis request latency** | < 100ms overhead (not counting model inference) |
| **Image upload processing** | Client-side resize to max 1024px before upload |

### 8.3 Image Optimization

```typescript
// Client-side before upload
function optimizeImage(file: File): Promise<Blob> {
  // 1. Load into canvas
  // 2. Resize if > 1024px on longest side
  // 3. Convert to JPEG quality 90
  // 4. Return optimized blob
  // This reduces upload time and server memory
}
```

---

## Appendix A: File Structure (Future Implementation)

```
SatQuery-AI/
├── satquery/                    # Existing Python backend (FROZEN)
│   ├── api.py                   # NEW: FastAPI wrapper
│   ├── pipeline.py
│   ├── router.py
│   ├── vlm.py
│   ├── sar_tool.py
│   ├── sar_infer.py
│   ├── visualize.py
│   └── demos.py
│
├── frontend/                    # NEW: React app
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── public/
│   │   └── favicon.svg
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/
│       │   └── client.ts            # API client functions
│       ├── types/
│       │   └── index.ts             # TypeScript interfaces
│       ├── hooks/
│       │   ├── useAnalyze.ts        # Analysis mutation hook
│       │   ├── useHistory.ts        # Query history state
│       │   └── useImageUpload.ts    # Image upload + optimize
│       ├── components/
│       │   ├── Header.tsx
│       │   ├── ImageUpload.tsx
│       │   ├── QueryInput.tsx
│       │   ├── AnalyzeButton.tsx
│       │   ├── ResultPanel.tsx
│       │   ├── VisualEvidence.tsx
│       │   ├── DetectionTable.tsx
│       │   ├── ConfidenceBar.tsx
│       │   ├── IntentBadge.tsx
│       │   ├── LoadingOverlay.tsx
│       │   ├── HistoryPanel.tsx
│       │   ├── DemoSelector.tsx
│       │   ├── AboutModal.tsx
│       │   └── Footer.tsx
│       ├── screens/
│       │   ├── LandingScreen.tsx
│       │   ├── WorkspaceEmpty.tsx
│       │   ├── WorkspaceWithImage.tsx
│       │   ├── AnalysisLoading.tsx
│       │   ├── OpticalResult.tsx
│       │   └── SARResult.tsx
│       ├── styles/
│       │   ├── globals.css
│       │   └── animations.css
│       └── lib/
│           ├── constants.ts
│           └── utils.ts
│
├── checkpoints/                  # Model weights (unchanged)
├── test_images/                  # Sample images (unchanged)
├── EarthDial/                    # EarthDial repo (unchanged)
├── sar_venv/                     # Isolated SAR env (unchanged)
├── LOOP_*_VALIDATION.md          # Loop reports
├── DEMO_SCRIPT.md
├── JUDGE_QA.md
├── CLAIMS_AND_LIMITATIONS.md
├── UI_UX_DESIGN_SPEC.md         # This document
├── README.md
└── requirements.txt
```

---

## Appendix B: Component Props Reference

```typescript
// Header
interface HeaderProps {
  onAbout: () => void;
}

// ImageUpload
interface ImageUploadProps {
  image: File | null;
  imagePreview: string;
  onUpload: (file: File) => void;
  onClear: () => void;
  isAnalyzing: boolean;
}

// QueryInput
interface QueryInputProps {
  query: string;
  onChange: (query: string) => void;
  onSubmit: () => void;
  suggestions: string[];
  disabled: boolean;
}

// ResultPanel
interface ResultPanelProps {
  result: AnalyzeResponse | null;
  isAnalyzing: boolean;
  selectedDemo: string;
}

// VisualEvidence
interface VisualEvidenceProps {
  annotatedImageUrl: string | null;
  originalImageUrl: string;
  intent: IntentType;
}

// DetectionTable
interface DetectionTableProps {
  detections: SARDetection[];
  type: 'sar' | 'grounding';
}

// IntentBadge
interface IntentBadgeProps {
  intent: IntentType;
  size?: 'sm' | 'md' | 'lg';
}

// LoadingOverlay
interface LoadingOverlayProps {
  elapsedSeconds: number;
  modelUsed: string;
  onCancel: () => void;
}

// HistoryPanel
interface HistoryPanelProps {
  entries: HistoryEntry[];
  onRevisit: (entry: HistoryEntry) => void;
  onClear: () => void;
}

// DemoSelector
interface DemoSelectorProps {
  demos: DemoScenario[];
  selected: string;
  onSelect: (name: string) => void;
}

// AboutModal
interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}
```

---

*End of design specification. Awaiting approval before React implementation begins.*
