if [ $# -ne 1 ]; then
	echo "usage: $0 path"
	exit -1
fi

cd $1

for directory in $(ls); do
	let n=$(ls -l $directory | wc -l)-1	# number of files

	if [ $n -gt 1 ]; then
		echo "$directory has $n files"
		let n2=$n-1

		for file in $(ls $directory); do
			rm $directory/$file
			echo "$file removed"
			let n2=$n2-1
			if [ $n2 -eq 0 ]; then
				break
			fi
		done

	elif [ $n -eq 0 ]; then
		rmdir $directory
		echo "$directory has 0 files"
		echo "$directory removed"
	elif [ $n -ne 1 ]; then
		echo "$directory has $n files"
	fi
done

echo done
