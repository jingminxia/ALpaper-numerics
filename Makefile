exec=mpiexec -n 22 python
pre:
	mkdir -p logs/

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

q0continuation: pre
	$(exec) q0continuation.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/q0continuation-almgstar.log
	$(exec) q0continuation.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/q0continuation-almgpbj.log

k2continuation: pre
	$(exec) k2continuation.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/k2continuation-almgstar.log
	$(exec) k2continuation.py --nref 1 --gamma 0 --k 2 --K 1 --solver-type almg-pbj --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/k2continuation-almgpbj.log

comptime: pre
	$(exec) periodic2d/periodic2d.py --nref 1 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref1.log || 1
	$(exec) periodic2d/periodic2d.py --nref 2 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref2.log || 1
	$(exec) periodic2d/periodic2d.py --nref 3 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref3.log || 1
	$(exec) periodic2d/periodic2d.py --nref 4 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref4.log || 1
	$(exec) periodic2d/periodic2d.py --nref 5 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref5.log || 1
	$(exec) periodic2d/periodic2d.py --nref 6 --gamma 1e6 --k 2 --K 1 --solver-type almg-star --stab-type continuous --prolong-type none --nonliear-iteration picard 2>&1 | tee logs/comptime-almgstar-ref6.log || 1

periodic2dallu: pre
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 0 2>&1 | tee logs/idealalgamma0.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1 2>&1 | tee logs/idealalgamma1.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1e1 2>&1 | tee logs/idealalgamma1e1.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1e2 2>&1 | tee logs/idealalgamma1e2.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1e3 2>&1 | tee logs/idealalgamma1e3.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1e4 2>&1 | tee logs/idealalgamma1e4.log || 1
