FROM jupyter/scipy-notebook:db3ee82ad08a
#FROM continuumio/miniconda3


#########
# Setup #
#########

ENV HOME /home/jovyan
WORKDIR $HOME

# Clone lcbg repo
RUN git clone https://github.com/crawfordsm/lcbg.git
ENV LCBGDIR $HOME/lcbg

# Clone kcorrect @ dcad853
RUN git clone https://github.com/blanton144/kcorrect.git
ENV KCORRECT_DIR $HOME/kcorrect

# Clone kcorrect_python @ 8ea0c62
RUN git clone https://github.com/nirinA/kcorrect_python.git
ENV KCORRECT_PYTHON_DIR $HOME/kcorrect_python


########################################
# Setup Conda and Install Requirements #
########################################

#RUN conda env update --file $LCBGDIR/environment.yml
RUN conda env create -f $LCBGDIR/environment.yml
ENV PATH /opt/conda/envs/lcbg/bin:$PATH
RUN echo "source activate lcbg" >> $HOME/.bashrc

ENV CONDA_DEFAULT_ENV lcbg


####################
# Install kcorrect #
####################

ENV PATH $KCORRECT_DIR/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$KCORRECT_DIR/lib
ENV IDL_PATH $KCORRECT_DIR/pro

WORKDIR $KCORRECT_DIR

RUN kevilmake -k

WORKDIR $HOME


###########################
# Install kcorrect_python #
###########################

WORKDIR $KCORRECT_PYTHON_DIR

RUN python setup.py install

WORKDIR $HOME


################
# Install lcbg #
################

WORKDIR $LCBGDIR

RUN python setup.py develop

WORKDIR $HOME

