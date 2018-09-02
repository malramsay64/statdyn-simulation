#
# Makefile
# Malcolm Ramsay, 2018-09-02 12:04
#

all:
	@echo "Makefile needs your attention"

build:
	docker build -t malramsay/sdrun .

test: build
	docker run malramsay/sdrun pytest

lint: build
	docker run malramsay/sdrun pylint src/
	docker run malramsay/sdrun mypy src/

.PHONY: test build

# vim:ft=make
#
