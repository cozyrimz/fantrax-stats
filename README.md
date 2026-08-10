# Fantrax custom-scoring analysis

Tools for a 12-team head-to-head EPL league with a custom scoring system across
50+ tracked statistics. The league rewards a wide range of actions so that most
player types carry value, and these scripts measure whether that is actually
happening and what to change if it is not.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Authentication uses your own browser session. Export your Fantrax cookies to
`cookies.json` in this directory, either as the array a cookie-export extension
produces or as `{"Cookie": "name=value; ..."}`.

## Getting the data

Fantrax has no public API for granular stats, and neither the Players page nor
its CSV export includes the tracked categories — both are limited to summary
columns. The per-game breakdown only exists on a player's Games (Fantasy) tab,
which the private `getPlayerProfile` endpoint serves one player at a time.

```bash
python3 fetch_player_index.py            # names, teams and roster status for the pool
python3 export_stats.py --recent 3       # per-game logs for the last three seasons
```

That endpoint is rate limited specifically for profile views, so the exporter
paces itself at 8 seconds per request, slows down further whenever Fantrax
pushes back, and retries a blocked player for up to ten minutes before moving
on. Raw responses are cached under `cache/{seasonId}/{playerId}.json`, so the
run resumes where it left off and re-running costs nothing.

Expect roughly 90 minutes per season for a 710-player pool.

## Running the analysis

```bash
python3 run_all.py                       # every season in the cache
python3 run_all.py --season 2025-26      # one season
```

This produces, for each season in the cache, under `output/seasons/<season>/`:

- `metrics_<season>.json` — every balance measurement
- `report.md` and `charts/` — the written diagnosis for that season
- `proposals/recommended-<season>.json` — the tuned weight set for that season

Shared artifacts stay in `output/`:

- `season_totals_<season>.csv` and `games_<season>.parquet` — the stat matrix
- `weights.json` and `weights.csv` — the league's scoring weights
- `archetypes.csv` — each player's derived playing style

`proposals/recommended.json` is a copy of the most recent season's recommendation.

### Recovering the scoring weights

Nothing needs to be transcribed from the league settings. Every game row
satisfies `FPts = sum(weight × stat)` for the player's position, so the weights
are the solution of a linear system that `recover_weights.py` solves per
position. The fit is exact, which also proves the stat parsing is correct. A few
categories cannot be observed because no player ever recorded the event; those
weights are unknowable from data but by definition cannot affect any score.

### What gets measured

Replacement level is the reference point, because whether waivers are worth
watching depends on the gap between a marginal starter and the best free agent.
The league fields 1 keeper, 3 to 5 defenders, 3 to 5 midfielders and 1 to 3
forwards from an 11-player lineup, so `league_config.py` derives how many
players at each position the league collectively has to start, and the player
ranked at that number sets replacement level.

On top of that: inequality of value over replacement, how many players score
within 80 percent of a median starter, the positional mix of the top 50 against
the mix of starting slots, and how much of each position's scoring comes from
high-volume categories that pay for minutes rather than skill.

A season-long curve says who was best over the year, but whether a waiver pickup
can help depends on whether the weekly leaderboard moves at all. So the per-game
logs are also grouped into weeks to measure how many different players reach a
weekly top thirty and what share of those places the season's twenty best
players occupy. If that share is high, the pool is static no matter how the
season-long curve looks.

## Testing a scoring change

Describe the change as a proposal and replay it over the real match data:

```bash
python3 simulate.py --proposal proposals/example.json
```

```json
{
  "name": "trim-volume",
  "description": "Cut the categories that pay for minutes",
  "scale": { "F": 1.15 },
  "multiply": { "all": { "BR": 0.5 }, "D": { "CLR": 0.6 } },
  "set": { "F": { "G": 12 } }
}
```

`all` applies to every position that scores the category, and `scale` multiplies
every category a position scores. Because scoring is linear, the resulting totals
are exactly what every player would have finished with, and the output compares
the distribution before and after along with the biggest risers and fallers.

### Why the two levers are separate

Cutting a category changes *which* players within a position have value. Scaling
a whole position cannot reorder the players inside it, so it only changes *how
many* of that position reach the top of the league. Keeping them separate is what
makes the problem tractable: the category cuts fix the mix of playing styles, and
then the position scalars can be solved for directly, by choosing the rank each
position deserves given how many the league must start and scaling so the player
at that rank is worth the same everywhere.

```bash
python3 sweep.py --season 2025-26        # how deep a cut costs what
python3 tune.py --base proposals/base-trim.json --out proposals/recommended.json
```

`sweep.py` traces the trade-off, because the volume categories are also a floor
that every starter collects: cutting them rebalances the positions but widens the
gap between the best players and the rest. `tune.py` solves the position scalars
for a given set of cuts, then rounds the result to a short list of round numbers
that can be typed into the league settings, keeping only the changes responsible
for most of the points moved and re-checking that the rounding changed nothing.

To search for a weight set rather than hand-tune one:

```bash
python3 optimize.py --iterations 2000 --restarts 6
```

The optimizer minimises a blend of value inequality, positional imbalance and
archetype concentration. Weights stay within a bounded multiple of their current
value and never change sign, and goals and assists can only go up.

## Interactive version

`build_canvas.py` writes a Cursor canvas with the data embedded, where weights
can be dragged and the value curve, positional mix and waiver depth update
immediately.

## Public web version (GitHub Pages)

`build_web.py` generates a self-contained `docs/index.html` from the same
embedded data as the canvas. It runs in any browser with no build step and is
suitable for GitHub Pages.

```bash
python3 build_web.py          # writes docs/index.html
python3 -m http.server -d docs 8080   # preview locally at http://localhost:8080
```

To publish:

1. Commit `docs/index.html` (regenerated whenever you run `run_all.py`).
2. In the GitHub repo, go to **Settings → Pages**.
3. Set **Source** to **Deploy from a branch**, branch **main**, folder **/docs**.
4. The site goes live at `https://<username>.github.io/fantrax-stats/` after a minute or two.

The page embeds player names and season totals from your league data. There are
no credentials, but treat the repo visibility accordingly.

## Files

| File | Purpose |
| --- | --- |
| `export_stats.py` | Fetches and caches per-game logs |
| `fetch_player_index.py` | Player names, teams and roster status |
| `fantrax_data.py` | Shared cache parsing and aggregation |
| `league_config.py` | Roster and lineup settings |
| `build_dataset.py` | Cache to stat matrices and season totals |
| `recover_weights.py` | Solves the scoring weights from the data |
| `archetypes.py` | Clusters playing styles within each position |
| `diagnose.py` | Balance metrics and replacement level |
| `report.py` | Charts and written report |
| `simulate.py` | Replays a proposed weight set |
| `sweep.py` | Traces the cost and benefit of cutting volume categories |
| `tune.py` | Solves position scalars and rounds the result for entry |
| `optimize.py` | Searches for a more even weight set |
| `build_canvas.py` | Generates the interactive Cursor canvas |
| `build_web.py` | Generates the public GitHub Pages scoring lab |
| `run_all.py` | Runs the pipeline end to end |
