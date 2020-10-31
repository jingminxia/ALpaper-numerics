exec=mpiexec -n 10 python
pre:
	mkdir -p logs/

q0continuation: pre
	python periodic2d/q0continuation.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/q0continuation-almgstar.log
	python periodic2d/q0continuation.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/q0continuation-almgpbj.log

k2continuation: pre
	python periodic2d/k2continuation.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/k2continuation-almgstar.log
	python periodic2d/k2continuation.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/k2continuation-almgpbj.log

newton: pre
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e3ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e3ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e3ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e3ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e3ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e4ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e4ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e4ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e4ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e4ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e5ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e5ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e5ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e5ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e5ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e6ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e6ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e6ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e6ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration newton 2>&1 | tee logs/newton-gamma1e6ref5.log || 1

picard: pre
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e3ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e3ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e3ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e3ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e3ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e4ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e4ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e4ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e4ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e4ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e5ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e5ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e5ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e5ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e5ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e6ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e6ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e6ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e6ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/picard-gamma1e6ref5.log || 1

comptime: pre
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 6 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref6.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e6 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgpbj-ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e6 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgpbj-ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e6 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgpbj-ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e6 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgpbj-ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e6 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgpbj-ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 6 --gamma 1e6 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgpbj-ref6.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e6 --k 2 --K 1 --solver-type mgvanka --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-mgvanka-ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e6 --k 2 --K 1 --solver-type mgvanka --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-mgvanka-ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e6 --k 2 --K 1 --solver-type mgvanka --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-mgvanka-ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e6 --k 2 --K 1 --solver-type mgvanka --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-mgvanka-ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e6 --k 2 --K 1 --solver-type mgvanka --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-mgvanka-ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 6 --gamma 1e6 --k 2 --K 1 --solver-type mgvanka --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-mgvanka-ref6.log || 1

alluperiodic2d: pre
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma0ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma0ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma0ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma0ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma0ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma10ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma10ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma10ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma10ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma10ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e2ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e2ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e2ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e2ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e2ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e3ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e3ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e3ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e3ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e3ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e4ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e4ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e4ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e4ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e4ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e5ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e5ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e5ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e5ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e5ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e6ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e6ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e6ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e6ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-periodic2d-gamma1e6ref5.log || 1

constaintimprovement: pre
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 10 --k 1 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/constraint-improvement-gamma10.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 50 --k 1 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/constraint-improvement-gamma50.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 1e2 --k 1 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/constraint-improvement-gamma1e2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 5e2 --k 1 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/constraint-improvement-gamma5e2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 1e3 --k 1 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/constraint-improvement-gamma1e3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 5e3 --k 1 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/constraint-improvement-gamma5e3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 1e4 --k 1 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/constraint-improvement-gamma1e4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 5e4 --k 1 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/constraint-improvement-gamma5e4.log || 1
	python oneconstant/error.py

alluoneconstant: pre
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma0ref1.log || 1
	$(exec) oneconstant/oneconstant.py --nref 2 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma0ref2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 3 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma0ref3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 4 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma0ref4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 5 --gamma 0 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma0ref5.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1ref1.log || 1
	$(exec) oneconstant/oneconstant.py --nref 2 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1ref2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 3 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1ref3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 4 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1ref4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 5 --gamma 1 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1ref5.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma10ref1.log || 1
	$(exec) oneconstant/oneconstant.py --nref 2 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma10ref2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 3 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma10ref3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 4 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma10ref4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 5 --gamma 10 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma10ref5.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e2ref1.log || 1
	$(exec) oneconstant/oneconstant.py --nref 2 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e2ref2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 3 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e2ref3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 4 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e2ref4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 5 --gamma 100 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e2ref5.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e3ref1.log || 1
	$(exec) oneconstant/oneconstant.py --nref 2 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e3ref2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 3 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e3ref3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 4 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e3ref4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 5 --gamma 1e3 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e3ref5.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e4ref1.log || 1
	$(exec) oneconstant/oneconstant.py --nref 2 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e4ref2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 3 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e4ref3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 4 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e4ref4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 5 --gamma 1e4 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e4ref5.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e5ref1.log || 1
	$(exec) oneconstant/oneconstant.py --nref 2 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e5ref2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 3 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e5ref3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 4 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e5ref4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 5 --gamma 1e5 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e5ref5.log || 1
	$(exec) oneconstant/oneconstant.py --nref 1 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e6ref1.log || 1
	$(exec) oneconstant/oneconstant.py --nref 2 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e6ref2.log || 1
	$(exec) oneconstant/oneconstant.py --nref 3 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e6ref3.log || 1
	$(exec) oneconstant/oneconstant.py --nref 4 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e6ref4.log || 1
	$(exec) oneconstant/oneconstant.py --nref 5 --gamma 1e6 --k 2 --K 1 --solver-type allu --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/allu-oneconstant-gamma1e6ref5.log || 1
