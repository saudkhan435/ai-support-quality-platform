import csv

# Read our original customer-support dataset
with open(
    "data/raw/support_tickets.csv",
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    # Find one password-recovery example
    for row in reader:

        if row["intent"] == "recover_password":

            print("Customer question:")
            print(row["instruction"])

            print("\nReference response:")
            print(row["response"])

            break