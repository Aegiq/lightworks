import datetime
import os

current_year = f"{datetime.datetime.now(tz=datetime.UTC).year:04d}"

for directory in ["lightworks", "docs", "tests"]:
    for root, _, files in os.walk(f"{directory}"):
        # Generate list of all python files in a directory
        py_files = [f for f in files if f.endswith(".py")]
        for f_name in py_files:
            with open(root + "/" + f_name, encoding="utf-8") as f:
                lines = f.readlines()
            # Loop through all lines until the copyright one is found, then edit
            # this and break the loop
            for i, line in enumerate(lines):
                if line.startswith("# Copyright"):
                    if line[16:22] != " - 202":
                        msg = (
                            "Copyright header may not be in the expected format"
                            f" for file {root + '/' + f_name}."
                        )
                        raise ValueError(msg)
                    lines[i] = line[:19] + current_year + line[23:]
                    continue
            # Write new python file
            with open(root + "/" + f_name, "w", encoding="utf-8") as f:
                f.writelines(lines)
