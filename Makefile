#
# Makefile
# Malcolm Ramsay, 2018-09-02 12:04
#

all:
	@echo "Makefile needs your attention"

build:
	docker build -t malramsay/sdrun .

format:
	black --py36 src
	black --py36 test

test: build
	docker run malramsay/sdrun pytest

llint:
	pylint src/
	mypy src/
	mypy test/

lint: build
	docker run malramsay/sdrun pylint src/
	,docker run malramsay/sdrun mypy src/

docs:
	$(MAKE) -C sphinx html

.PHONY: test build

# vim:ft=make
#
