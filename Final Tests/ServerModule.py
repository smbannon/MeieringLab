from flask import Flask, request, render_template
import csv
import pandas as pd

# useful stuff for linear regression
import numpy as np

# to get command-line arguments
import sys

# to work with files and directories
import os

# to manipulate iterators
from itertools import chain


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'

# The 'routes' are different urls
@app.route('/')
def main_page():
    return render_template('main.html')

# The form will submit to here
@app.route('/submit', methods=['POST'])
def submit_page():
   peaklists = request.form['Peaklists']
   references = request.form['Reference peaks (ppm)']
   temperatures = request.form['temperatures']
   start_index = request.form['Starting Temperature Index']
   assignments = request.files['Assignments']
   files = request.files.getlist("file[]")


   # processing the input data
   return render_template('output.html', output=4)
   #return MainProcess(files, references, temperatures, start_index, assignments)

def increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def decreasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# Function that operates on input data
def MainProcess(files, references, temperatures, start_index, assignments, outfile):
    temps = {}
    # 15N/1H Ξ (reference compound: liq. NH3)
    n_xi = 0.10132912

    # base transmitter frequencies (Hz)
    bf_h = 600130000
    bf_n = 60810645.0

    dist_cutoff = 0.25

    # chi squared cut-off for linearity; for best results keep this
    # relaxed (i.e. not too small).
    chi2_cutoff = 0.5

    # std dev cut-offs for point spacing in each dimension
    stdev_h_cutoff = 0.015
    stdev_n_cutoff = 0.100

    # largest spacing outlier allowed (in std devs; 5 is very generous)
    outlier_h_cutoff = 5
    outlier_n_cutoff = 5
    for i, fname in enumerate(files):
        df = pd.read_csv(fname)
        f1ppm = df.iloc[:, 6]
        f2ppm = df.iloc[:, 5]
        temps[fname] = list(zip(f1ppm, f2ppm))

    peaks = []
    infile = open(assignments.get().strip())
    for line in infile:
        splitline = line.split(",")
        # import peak list; name of assignment stored in last position of tuple
        peaks.append([splitline[1].strip(), splitline[2].strip(), splitline[0].strip()])

    sol_count = 0
    ass_count = 0
    # initialize empty dictionary (will contain temperature data)
    temps = {}
    # open output file handle
    outhandle = open(outfile.get(), "w")

    # write headers
    outlist = []
    outlist.append("Residue")
    outlist.append("Ass. 1H")
    outlist.append("Ass. 15N")
    outlist.append("Notes")
    outlist.append("1H del_sigma/del_T (ppb/K)")
    outlist.append("1H chi^2")
    outlist.append("15N del_sigma/del_T (ppb/K)")
    outlist.append("15N chi^2")
    outlist.append("del_sigma*N/del_T*H")
    outlist.append("chi^2")
    outlist.append("") # spacer
    for x,tempx in enumerate(temperatures.get().split(",")):
        outlist.append(tempx.strip()+"°C 1H")
        outlist.append(tempx.strip()+"°C 15N")
    outlist.append("") # spacer
    for x,tempx in enumerate(temperatures.get().split(",")):
        outlist.append(tempx.strip()+"°C 1H-RR")
        outlist.append(tempx.strip()+"°C 15N-RR")
    outhandle.write(",".join(outlist)+"\n")

    # get index of temperature/peak list to start with
    start = int(start_index.get()) - 1
    for peak in peaks:
        if peak[0] == "" or peak[1] == "":
            outlist = []
            outlist.append(peak[2])  # residue identifier
            outlist.append(str(peak[1]))  # 15N
            outlist.append(str(peak[0]))  # 1H
            outlist.append("")  # notes (blank)
            outhandle.write(",".join(outlist) + "\n")

        else:
            # find point in data from starting temperature that most closely matches
            # the assigned peak
            min_dist = 9999.99
            min_point = 9999
            count = 0
            for i, point in enumerate(temps[files[start]]):
                # dist is actually distance squared; calculating the square root gains
                # us nothing here
                dist = (point[0] - float(peak[0])) ** 2 + (point[1] - float(peak[1])) ** 2
                if dist < min_dist:
                    min_dist = dist
                    min_point = i
            curr = (temps[files[start]][min_point])
            # starting point (not really a line ye\t)
            lines = []
            lines.append([curr])

            # process remaining temperatures:
            #   Sarting point may be in the middle; go up from there first (appending
            #   points to candidate lines, then down after (prepending points).
            for k in chain(range(start + 1, len(files)), reversed(range(0, start))):

                oldlines = lines
                lines = []

                # find candidate lines: previous lines extended by a point
                # (within distance cut-off) from spectrum at the new temp
                for j, line in enumerate(oldlines):
                    # start from either the first or last point in line
                    if k > start:
                        curr = line[len(line) - 1]
                    else:
                        curr = line[0]

                    # calculate distances
                    for i, point in enumerate(temps[files[k]]):
                        # dist is actually distance squared; caculating the square root
                        # gains us nothing here
                        dist = (point[0] - curr[0]) ** 2 + (point[1] - curr[1]) ** 2
                        # if within cut-off, add to list
                        if dist < dist_cutoff:
                            if k > start:
                                lines.append(line + [point])
                            else:
                                lines.append([point] + line)

                oldlines = lines
                lines = []

                # assess linearity of candidate lines, discard obviously nonlinear
                #  (linear fitting after processing each new temp is somewhat
                #   inefficient, but probably preferable to allowing the number
                #   of lines under consideration to blow up)
                for line in oldlines:
                    # unpack x and y coordinates
                    n, h = zip(*line)

                    # calculate coeff. of determination
                    r2 = (np.corrcoef(h, n)[0, 1]) ** 2

                    # linear regression
                    p = np.polyfit(h, n, 1)

                    # calculate chi squared
                    chi2 = np.sum((np.polyval(p, h) - n) ** 2)

                    # keep if linear approximation is good enough
                    if chi2 < chi2_cutoff and (increasing(h) or decreasing(h)):
                        lines.append(line)


            # check out results
            i_best = 999
            chi2_min = 9999
            for i, line in enumerate(lines):
                # for std dev calculation
                ndiff = []
                hdiff = []
                for j in range(1, len(line)):
                    ndiff.append(line[j][0] - line[j - 1][0])
                    hdiff.append(line[j][1] - line[j - 1][1])

                    # calculate deviations from mean spacings
                hstdev = np.std(hdiff)
                hmean = np.mean(hdiff)
                hdiff_dm = []  # list of absolute differences from the mean

                for x, diff in enumerate(hdiff):
                    hdiff_dm.append(abs(diff - hmean))

                nstdev = np.std(ndiff)
                nmean = np.mean(ndiff)
                ndiff_dm = []  # list of absolute differences from the mean
                for x, diff in enumerate(ndiff):
                    ndiff_dm.append(abs(diff - nmean))

                # unpack x and y coordinates
                n, h = zip(*line)
                r2 = (np.corrcoef(h, n)[0, 1]) ** 2
                p = np.polyfit(h, n, 1)
                chi2 = np.sum((np.polyval(p, h) - n) ** 2)

                # if this set of points is more linear than the previous best,
                # and isn't weeded out by standard deviation cut-offs or
                # outlier checks, it becomes the leading candidate
                if chi2 < chi2_min and \
                        hstdev < stdev_h_cutoff and \
                        nstdev < stdev_n_cutoff and \
                        max(hdiff_dm) <= outlier_h_cutoff * hstdev and \
                        max(ndiff_dm) <= outlier_n_cutoff * nstdev:
                    best_i = i
                    chi2_min = chi2
                    r2_best = r2
                    std_best_h = hstdev
                    std_best_n = nstdev
                    hmaxdm = max(hdiff_dm)
                    nmaxdm = max(ndiff_dm)

                # if a good line was found, process and write output file
            if chi2_min != 9999:

                # unpack 1H and 15N coordinates
                n, h = zip(*lines[best_i])

                # rereference
                refs = references.get().split(",")
                h_rr = []
                for l, h_raw in enumerate(h):
                    h_rr.append(h_raw - float(refs[l]))

                n_rr = []
                for l, n_raw in enumerate(n):
                    # calculate DSS frequency in Hz
                    dss_freq = (float(refs[l]) * (bf_h / 1000000)) + bf_h
                    # multiply by Ξ ratio
                    n_zero_freq = dss_freq * n_xi
                    # find difference between calculated zero ppm and transmitter freq.
                    n_ppm_adj = (n_zero_freq - bf_n) / bf_n * 1000000
                    # rereference
                    n_rr.append(n_raw - n_ppm_adj)

                # calculate temperature coefficient using rereferenced shifts
                t = [float(ts) for ts in temperatures.get().split(",")]
                ph = np.polyfit(t, h_rr, 1)
                pn = np.polyfit(t, n_rr, 1)
                phn = np.polyfit(h_rr, n_rr, 1)

                # construct a line of output in list form
                outlist = []
                outlist.append(peak[2])  # residue identifier
                outlist.append(str(peak[1]))  # 1H
                outlist.append(str(peak[0]))  # 15N
                outlist.append("")  # notes (blank)
                outlist.append(str(ph[0] * 1000))  # 1H temp coefficient in ppb/K
                outlist.append(str(np.sum((np.polyval(ph, t) - h_rr) ** 2)))  # chi^2
                outlist.append(str(pn[0] * 1000))  # 15N temp coefficient in ppb/K
                outlist.append(str(np.sum((np.polyval(pn, t) - n_rr) ** 2)))  # chi^2
                outlist.append(str(phn[0]))  # slope in the 1H - 15N plane
                outlist.append(str(np.sum((np.polyval(phn, h_rr) - n_rr) ** 2)))  # chi^2

                outlist.append("")  # column spacer

                for l in range(len(h)):
                    outlist.append(str(h[l]))
                    outlist.append(str(n[l]))

                outlist.append("")  # column spacer

                for l in range(len(h_rr)):
                    outlist.append(str(h_rr[l]))
                    outlist.append(str(n_rr[l]))

                # output to file
                outhandle.write(",".join(outlist) + "\n")

                sol_count = sol_count + 1


            else:

                outlist = []
                outlist.append(peak[2])  # residue identifier
                outlist.append(str(peak[1]))  # 15N
                outlist.append(str(peak[0]))  # 1H
                outlist.append("No solution found.")  # notes
                outhandle.write(",".join(outlist) + "\n")


if __name__ == '__main__':
    app.run()