import argparse
import pandas as pd
import os
import random
import sys


def findCommonFolderFromEnd(string1, string2):
    elems1 = str.split(string1, "/")
    elems2 = str.split(string2, "/")
    for i in reversed(elems1):
        if i == "":
            continue
        for j in reversed(elems2):
            if i == j:
                return i
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change the dataroot from a csv/pkl dataset file')
    parser.add_argument('--newdataroot', type=str, default=None)
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--columnname', type=str, default="path")

    args = parser.parse_args()

    if args.newdataroot is None or args.file is None:
        raise ValueError("A dataset file and new dataroot string must be given")

    if ".pkl" in args.file:
        datasetfile = pd.read_pickle(args.file)
        ispkl = True
        iscsv = False

    elif ".csv" in args.file:
        datasetfile = pd.read_csv(args.file)
        ispkl = False
        iscsv = True

    else:
        raise ValueError("Dataset file type not recognized")

    print("Opened datasetfile {} with shape {}".format(args.file, datasetfile.shape))
    print(datasetfile.head())
    print("Identifying dataroot")

    currentdataroot = datasetfile.iloc[0][args.columnname]

    for i in range(100):
        path = datasetfile.iloc[random.randint(0, len(datasetfile.index))][args.columnname]
        currentdataroot = os.path.commonpath([path, currentdataroot])

    print("Current dataroot is {}".format(currentdataroot))

    commonfolder = findCommonFolderFromEnd(args.newdataroot, currentdataroot)
    computednewdataroot = os.path.join(args.newdataroot, currentdataroot.split("/"+commonfolder+"/")[1])

    print("Changing to {}, please confirm".format(computednewdataroot))

    # raw_input returns the empty string for "enter"
    yes = {'yes', 'y', 'ye', ''}
    no = {'no', 'n'}

    choice = input().lower()
    if choice in yes:
        datasetfile[args.columnname] = datasetfile[args.columnname].str.replace(currentdataroot, computednewdataroot)
        if iscsv:
            datasetfile.to_csv(args.file)
        if ispkl:
            datasetfile.to_pickle(args.file)
    elif choice in no:
        exit()
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")


