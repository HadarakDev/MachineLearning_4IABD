## add Norm at the end of Dir/Files
import os
from sys import path


def renameWithNorm(basePath):
    renameDirNorm(basePath)
    renameFilesNorm(basePath)

def renameDirNorm(basePath):
    dirs = os.listdir(basePath)
    for dir in dirs:
        if dir.find("norm") == -1:
            os.rename(basePath + dir, basePath + dir + "_norm")

def renameFilesNorm(basePath):
    dirs = os.listdir(basePath)
    for dir in dirs:
        files = os.listdir(basePath + dir)
        for file in files:
            if file.find("h5") != -1 and file.find("norm") == -1:
                tmpFile = file[0:-3] + "_norm.h5"
                os.rename(basePath + dir + "//" + file, basePath + dir + "//" + tmpFile)



## rename dir/file syntax
def renameSyntax(basePath):
    renameSyntaxDir(basePath)
    renameSyntaxFile(basePath)

def renameSyntaxDir(basePath):
    dirs = os.listdir(basePath)
    for dir in dirs:
        if dir.find("dropout") != -1:
            newDir = dir.replace("dropout_02", "dropout02")
            newDir = newDir.replace("dropout_01", "dropout01")
            if not path.exists(basePath + newDir):
                os.rename(basePath + dir, basePath + newDir)


def renameSyntaxFile(basePath):
    dirs = os.listdir(basePath)
    for dir in dirs:
        files = os.listdir(basePath + dir)
        for file in files:
            if file.find("dropout") != -1:
                newFile = file.replace("dropout_02", "dropout02")
                newFile = newFile.replace("dropout_01", "dropout01")
                if not path.exists(basePath + dir + "//" + newFile):
                    os.rename(basePath + dir + "//" + file, basePath + dir + "//" + newFile)