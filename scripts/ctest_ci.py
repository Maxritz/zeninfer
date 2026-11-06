#!/usr/bin/env python3
"""Run CTest and surface each failure as a GitHub Actions annotation.

The process keeps CTest's exit status. The JUnit report is temporary; the
ordinary CTest logs and --output-on-failure remain available to CI users.
"""
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def annotation_text(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main():
    with tempfile.TemporaryDirectory(prefix="zen-ctest-") as work:
        report = Path(work) / "ctest.xml"
        result = subprocess.run(["ctest", *sys.argv[1:], "--output-junit", str(report)])
        if result.returncode and report.is_file():
            try:
                root = ET.parse(report).getroot()
                for case in root.iter("testcase"):
                    failure = case.find("failure")
                    if failure is None:
                        failure = case.find("error")
                    if failure is None:
                        continue
                    name = case.get("name", "unknown test")
                    status = (failure.text or failure.get("message") or "Failed").strip()
                    output = (case.findtext("system-out") or "").strip()
                    details = status + ("\n" + output if output else "")
                    # Keep annotations bounded, but retain both the start and end
                    # of lengthy output where the failure message often appears.
                    if len(details) > 6000:
                        details = details[:3000] + "\n... output truncated ...\n" + details[-3000:]
                    message = annotation_text(f"{name}: {details}")
                    print(f"::error title=CTest failure::{message}", flush=True)
            except ET.ParseError as exc:
                print(f"::warning::Could not parse CTest JUnit report: {annotation_text(str(exc))}")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
