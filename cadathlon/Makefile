PYTHON ?= python3
.PHONY: all reference-test docs release clean
all:
	$(MAKE) -C pset/p1_benchmark/src all
	$(MAKE) -C pset/p2_bert/src all
	$(MAKE) -C pset/p3_cnn/src all
	$(MAKE) -C pset/p4_qwen/src all
	$(MAKE) -C pset/p5_wise/src all
reference-test:
	$(MAKE) -C pset/p1_benchmark/src reference-test
	$(MAKE) -C pset/p2_bert/src reference-test
	$(MAKE) -C pset/p3_cnn/src reference-test
	$(MAKE) -C pset/p4_qwen/src reference-test
	$(MAKE) -C pset/p5_wise/src reference-test
docs:
	$(PYTHON) common/render_docs.py
release:
	$(PYTHON) common/package.py release .
clean:
	$(MAKE) -C pset/p1_benchmark/src clean
	$(MAKE) -C pset/p2_bert/src clean
	$(MAKE) -C pset/p3_cnn/src clean
	$(MAKE) -C pset/p4_qwen/src clean
	$(MAKE) -C pset/p5_wise/src clean
