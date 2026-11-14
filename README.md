# SPHEREx Sky Viewer

Enter a sky position (name, degrees, or h:m:s) and watch SPHEREx images of that patch play like a short video.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload                           # open http://127.0.0.1:8000
pytest                                                  # offline tests (synthetic data)
python -m scripts.test_targets                          # live test on the challenge targets -> output/*.gif
```

## Embed / share by URL
`/?target=Barnard%27s%20Star&size=3&mode=diff&bin=2&det=0` or `/?ra=133.79&dec=-7.25&size=5`

## How it works
`data.py` finds images through IRSA (SIA2) and downloads small cutouts → `process.py` reprojects them onto one
north-up grid, optionally adds or subtracts them, applies one shared stretch → `main.py` serves frames to `static/index.html`.
New data needs no changes: every request queries the live archive. Set `SPHEREX_COLLECTION` if IRSA renames the collection.

## To verify on first live run
- Collection name (`Irsa.list_collections()`) and ObsCore column names (`t_min`, `access_url`).
- Which FLAGS bits to mask (currently any non-zero flag when "Hide flagged pixels" is on).
