import pyBigWig


# Path to your BigBed file
bigbed_file_path = "your_file.bigbed"

# Open the BigBed file
bigbed_file = pyBigWig.open(bigbed_file_path)

# Fetch data for a specific region (chromosome, start, end)
chromosome = "chr1"
start = 1000000
end = 2000000

# Get the values within the specified region
values = bigbed_file.values(chromosome, start, end)

# Print the values
print("Values within the region:", values)

# Close the BigBed file
bigbed_file.close()

