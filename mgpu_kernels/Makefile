ARCH ?= native
CONF ?= GV100

DEFAULT_BENCHES := peer_copy all_gather all_reduce
BENCHES ?= $(DEFAULT_BENCHES)

.PHONY: all clean list $(BENCHES)

all: $(BENCHES)

$(BENCHES):
	$(MAKE) -C $@ ARCH=$(ARCH) CONF=$(CONF) all

clean:
	@for d in $(BENCHES); do $(MAKE) -C $$d ARCH=$(ARCH) CONF=$(CONF) clean || true; done

list:
	@printf '%s\n' $(BENCHES)
