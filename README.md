The goal is to turn PlasticsEurope process database of LCI (Life cycle Inventory) - or any other LCI database into consolidated LCAs by using EU EF3.1 caracterization matrix and a python script.

The way it works =>

For each process file in xml in the root folder, the script extract the different flows UUID

The script then tries to match these flows UUID with the 90k+ flows in the flows folder as these flow files have the UUID directly as their filename

The script also tries name matching using multiple colums from the EF matrix

The script uses workers to attain 75% of the computer CPU and thus calculate everything faster (parallal processing).

It displays progress bars and saves the result directly as a xslx Excel files, as well as the statisctics of % of matched and unmatched flows.

The script also saves the unmatched flow to allow for manual review.
