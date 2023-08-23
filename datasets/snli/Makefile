PYTHON = python3
URL = http://nlp.stanford.edu/projects/snli/snli_1.0.zip
ZIPFILE = $(notdir $(URL))
DATADIR = $(basename $(ZIPFILE))

TARGETS = train.jsonl val.jsonl test.jsonl

all: $(TARGETS)

train.jsonl: conv.py label2id.json $(DATADIR)/$(DATADIR)_train.jsonl
	$(PYTHON) $< --labeldic $(filter-out $<,$^) --output $@

val.jsonl: conv.py label2id.json $(DATADIR)/$(DATADIR)_dev.jsonl
	$(PYTHON) $< --labeldic $(filter-out $<,$^) --output $@

test.jsonl: conv.py label2id.json $(DATADIR)/$(DATADIR)_test.jsonl
	$(PYTHON) $< --labeldic $(filter-out $<,$^) --output $@

$(DATADIR)/%.jsonl: $(DATADIR)

$(DATADIR): $(ZIPFILE)
	unzip $(ZIPFILE)
	touch -r $^ $@

$(ZIPFILE):
	wget $(URL)
