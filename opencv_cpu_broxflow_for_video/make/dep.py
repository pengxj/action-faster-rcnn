#!/usr/bin/env python
#
# Copyright (C) 2009 Alexander Kl"aser
# 
# This piece is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# 
# This software has been downloaded from:
# http://lear.inrialpes.fr/people/klaeser/software
#

import sys
import os
import string
import os.path
import re

HELP_USAGE = """
Usage: dep.py <link-target> <link-file> <build-dir> <root-dep-file> <include-dir> ...
"""

regSuffix = re.compile(r"\.[^.]*$")
regSrc = re.compile(r"^.*\.(c|cc|cpp)$")
regDep = re.compile(r"^.*\.d$")
regDepSplit = re.compile(r"\s*\\*\s*")

suffixes = ['.cpp', '.c', '.cc']
includeDirs = []


def parseDepFile(fileName):
	# read in the dependency file
	depFile = open(fileName, 'r')
	depStr = depFile.read()

	# discard everything up to the colon
	colonPos = depStr.find(":")
	assert colonPos > 0, "the dependency file '" + fileName + "' does not have the correct format"
	depStr = depStr[colonPos + 1:]

	# collect all included files
	return regDepSplit.split(depStr)


def findSourceFile(headerFile):
	# get the basename without extension
	headerFile = regSuffix.sub('', headerFile)
	if not headerFile:
		return None

	# iterate over known suffixes
	for suffix in suffixes:
		srcFile = headerFile + suffix

		# check whether a source file corresponding to the header exists
		if os.path.exists(srcFile):
			return srcFile
	
	# we add to the file path directory by directory and check whether it
	# exists in one of the include directories
	i = headerFile.find('/') + 1
	if i != 1:
		i = 0
	while True:
		# check whether a source file exists in one of the given include dirs
		for dir in includeDirs:
			# check all suffixes for source files
			for suffix in suffixes:
				srcFile = os.path.join(dir, headerFile[i:] + suffix)
				#srcFile = os.path.abspath(srcFile)
				if os.path.exists(srcFile):
					return srcFile

		# find next position of '/'
		i = headerFile.find('/', i) + 1
		if i <= 0:
			break

	return None


def main(argv):
	global includeDirs

	# check command line parameters
	if len(sys.argv) < 5:
		print HELP_USAGE
		return
	
	args = sys.argv
	args.pop(0)
	ruleTarget = args.pop(0)
	linkFile = args.pop(0)
	buildDir = args.pop(0)
	rootDepFile = args.pop(0)
	includeDirs = args


	# scan all dependency files for files we need to link to
	# do this recursively starting at the root dependency file
	linkFiles = set()
	incFiles = set()
	depFileStack = set([rootDepFile])
	depFilesDone = set()
	while depFileStack:
		# get the next dependency file to process from the stack
		depFile = depFileStack.pop()
		if depFile in depFilesDone:
			continue
		depFilesDone.add(depFile)

		# iterate over all source files in the dependency file
		for nextFile in parseDepFile(depFile):
			newDepFile = ""

			# if we have a source file, we need to link against it
			if regSrc.match(nextFile):
				linkFiles.add(nextFile)
				newDepFile = buildDir + "/" + regSuffix.sub(".d", nextFile)
			
			# check whether a .cpp/.c/.cc file exist
			srcFile = findSourceFile(nextFile)
			if srcFile != None:
				linkFiles.add(srcFile)
				newDepFile = buildDir + "/" + regSuffix.sub(".d", srcFile)

			# if the corresponding .d file exists as parameter, add it to the stack
			if newDepFile and os.path.exists(newDepFile):
				depFileStack.add(newDepFile)

	#
	# generate all necessary rules
	#

	# all includes of dependency files
	for i in linkFiles:
		i = regSuffix.sub(".d", i)
		print "-include " + buildDir + "/" + i
	print

	# dependencies for link file
	print linkFile + ": \\"
	for i in linkFiles:
		i = regSuffix.sub(".d", i)
		print "\t" + buildDir + "/" + i + " \\"
	print

	# print out all files we need to link against
	print ruleTarget + ": " + linkFile + " \\"
	for i in linkFiles:
		i = regSuffix.sub(".o", i)
		print "\t" + buildDir + "/" + i + " \\"
	print


if __name__ == "__main__":
	main( sys.argv )


