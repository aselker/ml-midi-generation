import os, os.path
def all_pklmidis(rootdir):
    midiFiles = []
    for subdir, ___, filename in os.walk(rootdir):
        for files in filename:
            if files.endswith(".mid"):
                midiFiles.append(os.path.join(subdir, files))
    return midiFiles

def unpkl(midis):
  files = []
  for i,l in enumerate(midis):
    print("Unpickling file {}.".format(i))
    files.append(pickle.load(open(l,'rb')))
  return files
