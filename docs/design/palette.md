# Bauhaus palette

Locked 2026-08-12; light values revised 2026-09-25. This file is the
source of truth for color. The same eight tokens are declared as
`PALETTE_LIGHT` / `PALETTE_DARK` in `src/gui/theme.py`. Do not invent extra
primaries. Do not use `#000000`.

**2026-09-25 light revision (tester: "way too bright, you can see every
line, dot, number and word").** Surfaces are off-white, not `#FFFFFF`.
Ink and hairlines are softer. Plot data uses a softer blue than the chrome.
Axis tick numbers use `plot_tick` and recede. Light strokes are no longer
heavier than dark. Overlaid ensembles fade with the square root of the
trace count (`plot_ensemble_alpha`). Matplotlib toolbars are drawn at 45 %
opacity. Dark mode is unchanged.

The encore mockup is a **layout** reference (Swiss header, hairline panes,
caption titles). Its older hard red `#e30613` and cool grays are retired.

## Tokens

```css
:root {
  --bg:      #F4F2EC;  /* warm paper, not white */
  --surface: #FBFAF7;  /* off-white; pure white read as glare */
  --ink:     #26241F;  /* warm black, never #000 — 14.9:1 */
  --muted:   #77736A;  /* 4.5:1, AA body text */
  --rule:    #E4E0D6;
  --red:     #C8322B;
  --yellow:  #E9B520;
  --blue:    #1B4E9B;
}

[data-theme="dark"] {
  --bg:      #1A1917;
  --surface: #232220;
  --ink:     #F2EFE6;
  --muted:   #A19C91;
  --rule:    #35332E;
  --red:     #E8564A;
  --yellow:  #F5C842;
  --blue:    #4A82D6;
}
```

## How they are used

| Token | Chrome | Plots |
|---|---|---|
| `bg` | Window, gutters, status bar | — |
| `surface` | Header, sidebar, plot panes | `plot_bg` |
| `ink` | Body text | Template / mean traces |
| `muted` | Secondary text, run meta | Ensemble shadow / RF background |
| `rule` | Hairlines, splitter, borders | Plot spines |
| `red` | — | Compare / noise |
| `yellow` | — | Firing-rate line, RF peak |
| `blue` | Brand, selected tab, selected row, primary buttons, focus ring | ACG, ISI, ensemble, RF target, scatter |

Yellow `#E9B520` on white is a **stroke color**, not 12px text (contrast
is ~1.9:1). Status / table text that must stay yellow uses `#8A6500`.
Green (`#1F7A4D` light / `#5DCAA0` dark) is a functional extra for
"good / clean" status only — it is not a Bauhaus primary.

A few derived stops exist only so 12px chrome stays WCAG AA on the
locked surfaces:

| Role | Why it is not the raw token |
|---|---|
| Dark `accent` fill `#1B4E9B` | `#4A82D6` + white is 3.85:1 |
| Dark plot blue `#6B9BE0` | `#4A82D6` on `#232220` is 4.13:1 |
| Dark noise text `#F28A82` | `#E8564A` on `#232220` is 4.44:1 |

## Rules

1. Ink is `#1B1B1B` in light mode. Never `#000000`.
2. Plot field is `surface`, not `bg`. Paper chrome frames white panes.
3. Selected chrome (tabs, tree/table row, checked buttons) is `blue` with
   white-on-blue, not a translucent wash that disappears.
4. Keep the existing semantic key names (`accent`, `plot_acg`, …). Remap
   them onto these tokens. Do not rename keys — QSS and `restyle_plots`
   break.
5. Command palette, experiment browser, undo, and the 1100×650 minimum
   stay parked.

## Header

The Swiss header is a 40px `setMenuWidget` strip, not a native menu bar:

`[■ ENCORE]  20251015A / chunk20 / kilosort4 · 312 cells    Standard STA …    [File] [Array] [Open run] [Light] [Population]`

Brand mark is an 8px blue square + `ENCORE` in Bauhaus blue.

## Light plot extras (2026-09-25)

| Name in `theme.py` | Value | Role | Contrast on surface |
|---|---|---|---|
| `_BLUE_PLOT_LIGHT` | `#4A72B8` | data blue (`plot_acg`, `plot_ensemble`, `plot_scatter`, …) | 4.6:1 |
| `_INK_PLOT_LIGHT` | `#3B3934` | mean traces, `plot_line` | 11:1 |
| `_TICK_LIGHT` (`plot_tick`) | `#948F85` | axis tick numbers only | 3.1:1 |

`text_tertiary` stays at `muted` (AA): panel captions use it.
