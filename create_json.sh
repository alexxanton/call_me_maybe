arg=$1
if [[ $arg != "" ]]; then
	touch data/input/${arg}s_definition.json
	touch data/input/${arg}_calling_tests.json
	vim data/input/${arg}s_definition.json
	vim data/input/${arg}_calling_tests.json
fi
