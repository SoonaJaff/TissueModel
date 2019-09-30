mainpath="C:\Users\Sona Jaff\Desktop\Report\Data"
if ~exist(mainpath,'dir')
    mkdir(mainpath);
end
original="C:\Users\Sona Jaff\Desktop\Report\original\raw";
if ~exist(original,'dir')
    mkdir(original);
end
resize="C:\Users\Sona Jaff\Desktop\Report\original\resize";
if ~exist(resize,'dir')
    mkdir(resize);
end
enhance="C:\Users\Sona Jaff\Desktop\Report\original\enhance";
if ~exist(resize,'dir')
    mkdir(resize);
end
for i=1:1:9
    filenames(i)="mdb00"+i+".pgm";
    imagename(i)=original+"\mdb00"+i+".jpg";
    resizename(i)=resize+"\mdb00"+i+".jpg";
end
for i=10:1:99
    filenames(i)="mdb0"+i+".pgm";
    imagename(i)=original+"\mdb0"+i+".jpg";
    resizename(i)=resize+"\mdb0"+i+".jpg";
end
for i=100:1:322
    filenames(i)="mdb"+i+".pgm";
    imagename(i)=original+"\mdb"+i+".jpg";
    resizename(i)=resize+"\mdb"+i+".jpg";
end
for j=1:322
    
    img=imread(filenames(j));
    %final=rmbackground(img);
    final=imresize(img,[224,224]);
    imwrite(final,resizename(j));
    imwrite(img,imagename(j));
end
