import pandas as pd
import os

def count_subdomains(domain: str) -> int:
    if not isinstance(domain, str) or domain.strip() == "":
        return 0
    # strip leading/trailing dots and split
    parts = domain.strip(".").split(".")
    # at least 2 parts are the main domain + tld, everything before that is subdomain
    return max(len(parts) - 2, 0)

def main():
    # read input file
    filename = os.path.join("datasets", "urldata.csv")
    new_filename = os.path.join("datasets", "urldata_subdomains.csv")
    df = pd.read_csv(filename)

    # check column exists
    if "Domain" not in df.columns:
        raise ValueError("Expected a column named 'Domain' in urldata.csv")

    # compute subdomain count
    df["Subdomain_Count"] = df["Domain"].apply(count_subdomains)

    # save new file
    df.to_csv(new_filename, index=False)
    print("Saved output to /datasets/urldata_with_subdomains.csv")

if __name__ == "__main__":
    main()
