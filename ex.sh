###########################
# Final step
###########################

nohup mitosis data sindy_fe \
    --debug \
    -F trials/dynamics \
    -p data.filename=poly-864 \
    -e sindy_fe.seed=1 \
    -p sindy_fe.diff_params=x-smooth \
    -e sindy_fe.whitening=False \
    -p sindy_fe.reg_mode=trap-merge-dindy \
    -p sindy_fe.ens_kwargs=old-default &> sindy-poly-864.out &

nohup mitosis data sindy_fe \
    --debug \
    -F trials/dynamics \
    -p data.filename=pinv-864 \
    -e sindy_fe.seed=1 \
    -p sindy_fe.diff_params=x-smooth \
    -e sindy_fe.whitening=False \
    -p sindy_fe.reg_mode=trap-merge-dindy \
    -p sindy_fe.ens_kwargs=old-default &> sindy-pinv-864.out &

nohup mitosis data sindy_fe \
    --debug \
    -F trials/dynamics \
    -p data.filename=lin-864 \
    -e sindy_fe.seed=1 \
    -p sindy_fe.diff_params=x-smooth \
    -e sindy_fe.whitening=False \
    -p sindy_fe.reg_mode=trap-merge-dindy \
    -p sindy_fe.ens_kwargs=old-default &> sindy-lin-864.out &

nohup mitosis data sindy_fe \
    --debug \
    -F trials/dynamics \
    -p data.filename=ppara-864 \
    -e sindy_fe.seed=1 \
    -p sindy_fe.diff_params=x-smooth \
    -e sindy_fe.whitening=False \
    -p sindy_fe.reg_mode=trap-merge-dindy \
    -p sindy_fe.ens_kwargs=old-default &> sindy-para-864.out &

nohup mitosis data hankel_fe \
    --debug \
    -F trials/dynamics \
    -p data.filename=poly-864 \
    -p hankel_fe.diff_params=x-smooth \
    -p hankel_fe.variance=default \
    -p hankel_fe.hankel_kwargs=default &> hankel-poly-864.out &

nohup mitosis data hankel_fe \
    --debug \
    -F trials/dynamics \
    -p data.filename=ppara-864 \
    -p hankel_fe.diff_params=x-smooth \
    -p hankel_fe.variance=default \
    -p hankel_fe.hankel_kwargs=default &> hankel-para-864.out &

nohup mitosis data hankel_fe \
    --debug \
    -F trials/dynamics \
    -p data.filename=lin-864 \
    -p hankel_fe.diff_params=x-smooth \
    -p hankel_fe.variance=default \
    -p hankel_fe.hankel_kwargs=default &> hankel-lin-864.out &

nohup mitosis data hankel_fe \
    --debug \
    -F trials/dynamics \
    -p data.filename=pinv-862 \
    -p hankel_fe.diff_params=x-smooth \
    -p hankel_fe.variance=default \
    -p hankel_fe.hankel_kwargs=default &> hankel-pinv-862.out &

###########################
# Center regression
###########################

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=862pts \
    -e multireg_center.r_split=1000 &> 862.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=864pts \
    -e multireg_center.r_split=1000 &> 864.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=865pts \
    -e multireg_center.r_split=1000 &> 865.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=866pts \
    -e multireg_center.r_split=1000 &> 866.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=867pts \
    -e multireg_center.r_split=1000 &> 867.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=868pts \
    -e multireg_center.r_split=1000 &> 868.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=869pts \
    -e multireg_center.r_split=1000 &> 869.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=871pts \
    -e multireg_center.r_split=1000 &> 871.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=913pts \
    -e multireg_center.r_split=1000 &> 913.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=914pts \
    -e multireg_center.r_split=1000 &> 914.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=916pts \
    -e multireg_center.r_split=1000 &> 916.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=917pts \
    -e multireg_center.r_split=1000 &> 917.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=919pts \
    -e multireg_center.r_split=1000 &> 919.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=920pts \
    -e multireg_center.r_split=1000 &> 920.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=lo1pts \
    -e multireg_center.r_split=1000 &> lo1.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=windpts \
    -e multireg_center.r_split=1000 &> wind.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=nowindpts \
    -e multireg_center.r_split=1000 &> nowind.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=hi1pts \
    -e multireg_center.r_split=1000 &> hi1.out &

nohup mitosis data multireg_center \
    -F trials/center-regress \
    -p data.filename=hi2pts \
    -e multireg_center.r_split=1000 &> hi2.out &

###########################
# Bad Centers
###########################
nohup mitosis plume_points \
    --debug \
    -F trials/center \
    -p plume_points.filename=low-866 \
    -p plume_points.img_range=866 \
    -p plume_points.fixed_range=866 \
    -p plume_points.gauss_space_kws=high \
    -p plume_points.gauss_time_kws=high \
    -p plume_points.contour_kws=3c  &> 866-high.out &

nohup mitosis plume_points \
    --debug \
    -F trials/center \
    -p plume_points.filename=low-866 \
    -p plume_points.img_range=866 \
    -p plume_points.fixed_range=866 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=1c  &> 866-1c.out &


###########################
# Centerpoint extraction
###########################
nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-862 \
    -p plume_points.img_range=862 \
    -p plume_points.fixed_range=862 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 862-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-864 \
    -p plume_points.img_range=864 \
    -p plume_points.fixed_range=864 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 864-ideal.out &


nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-865 \
    -p plume_points.img_range=865 \
    -p plume_points.fixed_range=865 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 865-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-866 \
    -p plume_points.img_range=866 \
    -p plume_points.fixed_range=866 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 866-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-867 \
    -p plume_points.img_range=867 \
    -p plume_points.fixed_range=867 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 867-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-868 \
    -p plume_points.img_range=868 \
    -p plume_points.fixed_range=868 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 868-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-869 \
    -p plume_points.img_range=869 \
    -p plume_points.fixed_range=869 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 869-ideal.out &


nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-913 \
    -p plume_points.img_range=913 \
    -p plume_points.fixed_range=913 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 913-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=med-914 \
    -p plume_points.img_range=914 \
    -p plume_points.fixed_range=914 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 914-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=med-916 \
    -p plume_points.img_range=916 \
    -p plume_points.fixed_range=916 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 916-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=med-917 \
    -p plume_points.img_range=917 \
    -p plume_points.fixed_range=917 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 917-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=med-871 \
    -p plume_points.img_range=871 \
    -p plume_points.fixed_range=871 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c  &> 871-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=hi-919 \
    -p plume_points.img_range=919 \
    -p plume_points.fixed_range=919 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c &> 919-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=hi-920 \
    -p plume_points.img_range=920 \
    -p plume_points.fixed_range=920 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c &> 920-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=low-1 \
    -p plume_points.img_range=lo1 \
    -p plume_points.fixed_range=lo1 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c &> lo1-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=hi-1 \
    -p plume_points.img_range=hi1 \
    -p plume_points.fixed_range=hi1 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c &> hi1-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=hi-2 \
    -p plume_points.img_range=hi2 \
    -p plume_points.fixed_range=hi2 \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c &> hi2-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=blender-nowind \
    -p plume_points.img_range=nowind \
    -p plume_points.fixed_range=nowind \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.contour_kws=3c &> nowind-ideal.out &

nohup mitosis plume_points \
    -F trials/center \
    -p plume_points.filename=blender-wind \
    -p plume_points.img_range=wind \
    -p plume_points.fixed_range=wind \
    -p plume_points.gauss_space_kws=low \
    -p plume_points.gauss_time_kws=off \
    -p plume_points.circle_kw=80circ \
    -p plume_points.contour_kws=3c &> wind-ideal.out &

