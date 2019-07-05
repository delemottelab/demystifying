material change ambient AOChalky 0.1
material change outline AOChalky 1.4
display depthcue off
display rendermode GLSL
display update on
axes location off
display projection orthographic
color Display Background white
display shadows on
display ambientocclusion on


proc make_trajectory_movie_files {} {
	set num [molinfo top get numframes]
  puts "rendering trajectory with $num frames"
  set molname "holo"
	# loop through the frames
  set counter 0
  #Iterating backwards below
	for {set i $num} {$i > -1 } {incr i -1} {
		# go to the given frame
		animate goto $i
    # force display update
    display update 
    #take picture of frame
    set outfile [format "videooutput/$molname/frame.%05d.ppm" $counter]
    puts "Rendering frame $i to file $outfile"
    render TachyonLOptiXInternal $outfile 
    incr counter
  }
  puts "To render run the following command in your terminal: "
  #puts "to convert to AVI format, run: 'ffmpeg -i input.mp4 -c:v libx264 -c:a libmp3lame -b:a 384K output.avi'"
  puts "ffmpeg -framerate 60 -pattern_type glob -i \"videooutput/$molname/frame.*.ppm\" -c:v libx264 -preset slow -crf 18 -r 60 videooutput/$molname.mp4"
}
make_trajectory_movie_files