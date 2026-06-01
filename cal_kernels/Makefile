ARCH ?= native
CONF ?= GV100

DEFAULT_BENCHES := \
  VAL_l2_parallel \
  VAL_hbm_parallel \
  REAL_gather \
  REAL_gemm \
  REAL_reduce \
  REAL_scatter \
  REAL_softmax \
  REAL_spmm \
  REAL_vadd

BENCHES ?= $(DEFAULT_BENCHES)

.PHONY: all clean list $(BENCHES)

all: $(BENCHES)

$(BENCHES):
	$(MAKE) -C $@ ARCH=$(ARCH) CONF=$(CONF) all

clean:
	@for d in $(BENCHES); do $(MAKE) -C $$d ARCH=$(ARCH) CONF=$(CONF) clean || true; done

list:
	@printf '%s\n' $(BENCHES)
