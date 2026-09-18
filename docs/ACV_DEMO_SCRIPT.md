# ACV demo — team purple

Record the live app in a browser, with the cursor visible. Target 90–120 seconds; keep the final video below three minutes. Do not show terminals, credentials, or unrelated tabs. Save the finished MP4 as `outputs/acv/demo_video.mp4`, then run `python scripts/package_submission.py --team purple`.

| Time | Action and narration |
|---|---|
| 0–15 s | Show the app title. “This is purple's ACV fault-localisation app. It compares cooling behaviour across cars and produces a complete inspection order.” |
| 15–35 s | Upload `acv_test_case.xlsx` and click **Analyze uploaded files**. “Upload a telemetry workbook. Parsing, validity checks and comparative features run locally.” |
| 35–55 s | Show the ranking and evidence table. “All eight cars remain in the ranking. These scores compare evidence; they are not probabilities of refrigerant leakage.” |
| 55–75 s | Open **Cooling evidence** and compare the leading cars. “The plots help an engineer inspect sustained cooling differences and missing intervals.” |
| 75–95 s | Open **Data quality**. “Missing and invalid readings are separated. Nested development validation uses six independent cases, so hidden-test performance remains unknown.” |
| 95–115 s | Return to **Car ranking** and download **predictions.zip**. “The app exports the required CSV inside a ZIP, using the same validated inference service as the CLI.” |

If time permits, click **Check ranking stability** and describe the result as an hourly evidence stability diagnostic. Do not call it a calibrated confidence interval or fault probability. Leave enough time to show the download.

The repository also includes `scripts/record_demo.cjs`, a user-run Playwright capture script. It needs the Node.js Playwright package and Chromium, and a live app on port 8501. Its WebM output must be converted to MP4 and checked for duration before packaging. This script has not been executed by the implementation agent; it is an optional recording aid.
