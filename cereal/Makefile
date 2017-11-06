
downloaded_deps/cereal.tar.gz :
	mkdir -p downloaded_deps
	cd downloaded_deps && \
	curl -L -o cereal.tar.gz https://github.com/USCiLab/cereal/archive/v1.2.2.tar.gz

downloaded_deps/cereal : downloaded_deps/cereal.tar.gz
	cd downloaded_deps && \
	tar xf cereal.tar.gz && \
	ln -s cereal-1.2.2 cereal

clean:
	rm -Rf downloaded_deps
