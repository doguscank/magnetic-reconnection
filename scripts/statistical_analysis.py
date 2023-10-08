import matplotlib.pyplot as plt
from data_utils import *


# Function to filter data based on conditions
def filter_data(df):
    neg_bz = df["BZ, nT (GSM)"] < 0
    flow_speed_thr = (
        df["SW Plasma Speed, kms"].mean() + 2.0 * df["SW Plasma Speed, kms"].std()
    )
    high_plasma_V = df["SW Plasma Speed, kms"] > flow_speed_thr
    negative_dst_idx = df["Dst-index, nT"] < 0

    return neg_bz, high_plasma_V, negative_dst_idx


# Function to calculate match rates
def calculate_match_rates(
    year, neg_bz, high_plasma_V, negative_dst_idx, total_len, data_dict: dict
):
    num_items_matched = sum(neg_bz & high_plasma_V)
    num_items_matched_dst = sum(neg_bz & negative_dst_idx)
    match_indices = sum(neg_bz & high_plasma_V & negative_dst_idx)

    data_dict = {
        **data_dict,
        year: {
            "Total length": total_len,
            "# of items matched": match_indices,
            "# of negative B_z components": sum(neg_bz),
            "# of high plasma V components": sum(high_plasma_V),
            "# of negative Dst index components": sum(negative_dst_idx),
            "High plasma speed match rate": num_items_matched / sum(neg_bz) * 100
            if sum(neg_bz) != 0
            else 0,
            "Negative Dst index match rate": num_items_matched_dst / sum(neg_bz) * 100
            if sum(neg_bz) != 0
            else 0,
            "Total match rate": match_indices / sum(neg_bz) * 100
            if sum(neg_bz) != 0
            else 0,
            "Rate of occurrence": match_indices / total_len * 100,
        },
    }

    return data_dict


# Reading and preparing data
drop_columns = [
    "Proton flux (>1 Mev)",
    "Proton flux (>2 Mev)",
    "Proton flux (>4 Mev)",
]

data_dict = {}

for year in range(1963, 2024):
    df = read_data(year, drop_columns=drop_columns)
    selected_columns = [
        "Scalar B, nT",
        "Vector B Magnitude,nT",
        "Lat. Angle of B (GSE)",
        "Long. Angle of B (GSE)",
        "BX, nT (GSE, GSM)",
        "BY, nT (GSE)",
        "BZ, nT (GSE)",
        "BY, nT (GSM)",
        "BZ, nT (GSM)",
        "SW Plasma Speed, kms",
        "SW Plasma flow long. angle",
        "SW Plasma flow lat. angle",
        "E elecrtric field",
        "Plasma Beta",
        "Alfen mach number",
        "Dst-index, nT",
    ]

    total_len = len(df)

    drop_na(df)
    df = set_fillers_nan(df)

    df = df[selected_columns]

    # Filtering data
    neg_bz, high_plasma_V, negative_dst_idx = filter_data(df)

    # Calculating match rates
    data_dict = calculate_match_rates(
        year, neg_bz, high_plasma_V, negative_dst_idx, total_len, data_dict
    )

with open("data_dict.json", "w") as f:
    json.dump(data_dict, f)

freq_data = [data_dict[year]["Rate of occurrence"] for year in data_dict.keys()]

plt.bar(list(data_dict.keys()), freq_data)
plt.title("Hourly Rate of Occurance of Magnetic Reconnection")
plt.xticks(list(data_dict.keys()), list(data_dict.keys()), rotation=90)
plt.ylabel("Rate of Occurance (%)")
plt.xlabel("Year")
plt.show()
