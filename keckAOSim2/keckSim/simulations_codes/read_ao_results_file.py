import re
from collections import defaultdict

# Dictionary of system aliases
SYSTEM_ALIASES = {
    "sh56": "SH 56x56",
    "56x56": "SH 56x56",
    "haka56": "SH 56x56",

    "sh28": "SH 28x28",
    "28x28": "SH 28x28",
    "haka28": "SH 28x28",

    "sh20": "SH 20x20",
    "20x20": "SH 20x20",
    "xin": "SH 20x20",
    "xinetics": "SH 20x20",
}

def normalize_system(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    for alias, standard in SYSTEM_ALIASES.items():
        if key.startswith(alias):
            return standard
    raise ValueError(f"Unknown system alias: {name}")

def parse_text_file(filename: str):
    """
    Parses the AO study results text file into a nested dict structure:
    data[system][study_type] = list of dicts with r0, mag, best, sr
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    data = defaultdict(lambda: defaultdict(list))
    system, study_type = None, None

    for line in lines:
        # Detect block headers
        if "closed-loop" in line.lower():
            if "frequency" in line.lower():
                study_type = "frequency"
            elif "gain" in line.lower():
                study_type = "gain"

            # Try to also catch system if it's on the same line
            if "56x56" in line:
                system = "SH 56x56"
            elif "28x28" in line:
                system = "SH 28x28"
            elif "20x20" in line or "xinetics" in line.lower():
                system = "SH 20x20"

        # Detect lines that define the system alone
        elif re.match(r"^(SH\s*\d+x\d+|xinetics)", line, re.IGNORECASE):
            if "56x56" in line:
                system = "SH 56x56"
            elif "28x28" in line:
                system = "SH 28x28"
            elif "20x20" in line or "xinetics" in line.lower():
                system = "SH 20x20"

        # Detect table rows
        elif re.match(r"^\d+", line):
            parts = line.split()
            if study_type == "frequency":
                r0, mag, best, sr = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
                data[system][study_type].append(
                    {"r0": r0, "mag": mag, "Best_Frequency": best, "Max_SR": sr}
                )
            elif study_type == "gain":
                r0, mag, best, sr = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
                data[system][study_type].append(
                    {"r0": r0, "mag": mag, "Best_Gain": best, "Max_SR": sr}
                )

    return data

def get_best_value(data, system_name, r0, mag, study_type):
    system = normalize_system(system_name)
    if system not in data or study_type not in data[system]:
        raise ValueError(f"No data for {system} {study_type}")

    for row in data[system][study_type]:
        if row["r0"] == r0 and row["mag"] == mag:
            return row
    raise ValueError(f"No entry for r0={r0}, mag={mag} in {system} {study_type}")


# Example usage:
if __name__ == "__main__":
    path = '/home/mcisse/keckAOSim/keckSim/data/'
    filename = f"{path}summary_results.txt"   # your text file
    data = parse_text_file(filename)

    result = get_best_value(data, "Xin", 16, 12, "frequency")
    print("Best result:", result)

