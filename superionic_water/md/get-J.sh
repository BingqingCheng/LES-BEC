prefix=$1
for i in {0..9}; do 
	il=$(echo "$i*40000" | bc); fl=$(echo "($i+1)*40000" | bc); echo $il $fl; 
	mv q-nvt-${prefix}.xyz-$il-$fl q-nvt-${prefix}-$il-$fl.xyz; 
	python ./get-J.py q-nvt-$prefix-$il-$fl.xyz; wait; 
done
for i in {0..9}; do il=$(echo "$i*40000" | bc); fl=$(echo "($i+1)*40000" | bc); cat q-nvt-${prefix}-$il-$fl.xyz_Jt.dat; done > q-nvt-${prefix}_Jt.dat
for i in {0..9}; do il=$(echo "$i*40000" | bc); fl=$(echo "($i+1)*40000" | bc); cat q-nvt-${prefix}-$il-$fl.xyz_J_topo_t.dat; done > q-nvt-${prefix}_J_topo_t.dat
