# Public repository release checklist

## Required before DOI/publication
- [ ] Exact source-dataset access URL inserted.
- [ ] Code license selected by authors.
- [ ] Derived-data/results licensing statement selected by authors.
- [ ] Repository name and version confirmed.
- [ ] Package validation script returns PASS.
- [ ] SHA-256 manifest frozen.
- [ ] No credentials, tokens, private Drive links, or `kaggle.json` included.
- [ ] Optional large-artifact deposit decision documented.
- [ ] Zenodo/OSF DOI obtained.
- [ ] Manuscript Data Availability and Code Availability statements updated with DOI.

## Recommended GitHub + Zenodo workflow
1. Create a public GitHub repository from this folder.
2. Add an explicit license selected by the authors.
3. Tag the release as `v2.0.0`.
4. Connect/import the tagged release to Zenodo and mint a DOI.
5. Cite the version-specific DOI in the manuscript.
6. Use future version numbers for any later corrections; do not silently replace the archived release.
