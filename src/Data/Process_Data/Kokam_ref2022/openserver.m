function [ model ] = openserver(  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
   comsolPort=[3036 3037 3038 3039 3040 3041 3042 3043 3044 3045 3046 3047 3048 3049 3050 3051 3052 3053 3054 3055];

   g = getCurrentTask();
   labit=g.ID;

   import com.comsol.model.*
   import com.comsol.model.util.*

   model = ModelUtil.model(num2str(comsolPort(labit)));

end

