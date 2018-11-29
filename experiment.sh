constdata="False"
constinit="False"
constdrop="True"
for i in {1..5}
do
  python modelnet/train.py --classes 40 --dataroot /workspace/ModelNet/modelnet40-normal_numpy/ --checkpoints_dir /workspace/ModelNet/SO-Net/SO-Net/checkpoints/model_const_Dt${constdata}_I${constinit}_Dr${constdrop}_${i} --const_traindata ${constdata} --const_weightinit ${constinit} --const_droporder ${constdrop}
done

for i in {0..9}
do
  python modelnet/train.py --classes 40 --dataroot /workspace/ModelNet/modelnet40-normal_numpy/ --checkpoints_dir /workspace/ModelNet/SO-Net/SO-Net/checkpoints/model_bagging_${i} --subset_suffix _bagging_v${i}
done

#for constdata in "True" "False"
#do
#  for constinit in "True" "False"
#  do
#    for constdrop in "True" "False"
#    do
#      if [ "$constdata" != "False" ] || [ "$constdrop" != "False" ]
#      then
#        for i in {3..5}
#        do
#          python modelnet/train.py --classes 40 --dataroot /workspace/ModelNet/modelnet40-normal_numpy/ --checkpoints_dir /workspace/ModelNet/SO-Net/SO-Net/checkpoints/model_const_Dt${constdata}_I${constinit}_Dr${constdrop}_${i} --const_traindata ${constdata} --const_weightinit ${constinit} --const_droporder ${constdrop}
#        done
#      fi
#    done
#  done
#done

