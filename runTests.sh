for i in {0..9};
do
./pointTrans2 -t 3000000 > cuda_mm_${i}.txt;
done;
python average.py 10 > average.txt;
