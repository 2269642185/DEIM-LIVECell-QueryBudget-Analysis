# Manual GitHub release instructions

Recommended repository name: `DEIM-LIVECell-QueryBudget-Analysis`.
Alternative: `LIVECell-QueryBudget-Audit`.

1. Manually create a new **PUBLIC** GitHub repository with the selected name. Use a clean repository; do not publish the private development repository or its Git history.
2. Upload only the contents of `reproducibility_archive_release/` at the repository root. Do not upload the enclosing submission package, validation work directories, manuscripts, private reports, raw logs or omitted artifacts. Avoid generating a second license/README during repository creation.
3. Verify the uploaded root LICENSE is Apache-2.0 and README/THIRD_PARTY_NOTICES render correctly. Check the uploaded file inventory and SHA-256 values against MANIFEST_SHA256.txt; the manifest excludes itself.
4. Create a tagged release such as `v1.0-submission` for the verified commit. Optionally attach the final candidate ZIP and its external `.zip.sha256` sidecar as release assets. The manually supplied ZIP has a different container layout/hash from GitHub's automatically generated source archive.
5. Record the actual public repository URL and tagged-release URL after publication. No URL/DOI placeholder is present in CITATION.cff. Add a real repository-code field only when the repository exists.
6. Supply the verified public URL/DOI for a separately authorized final Data Availability edit. If any tracked archive file changes, regenerate the manifest and release ZIP/checksum and verify them before tagging.

Official GitHub release instructions: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository

Nothing was created or uploaded by this preparation step. Reproduction limits remain: selected derived tables/figures YES; full prediction audit, full inference and full historical training from the public archive alone NO.
