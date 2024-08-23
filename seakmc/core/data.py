# coding: utf-8
# Copyright (c) SEAKMC Team.
# Distributed under the terms of the MIT License
import os, warnings
import numpy as np
import pandas as pd
import copy
from numpy import pi
from typing import Optional, Tuple
from monty.json import MSONable

from pymatgen.symmetry.analyzer import (
    PointGroupAnalyzer,
    SpacegroupAnalyzer,
)

from pymatgen.io.lammps.data import ATOMS_HEADERS
from pymatgen.io.lammps.data import LammpsBox, LammpsData
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Molecule, Structure

from seakmc.input.Input import Global_Variables
from seakmc.core.util import mat_lengths, mat_angles, mat_mag, generate_rotation_matrix
from seakmc.core.util import to_half_matrix

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

DispSettKEY = ["Selection", "Range", "Operation", "Values"]
EntryKEY = ["type", "id", "x", "y", "z", "dxy", "dxz", "dyz", "dxyz"]
cell_neigh_array = np.array([
    [0, 0, 0], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1],
    [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
    [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1],
])


class SeakmcBox(LammpsBox, MSONable):
    def __init__(
            self,
            bounds,
            tilt=None,
    ):
        if not tilt: tilt = np.zeros(3, dtype=float)

        super().__init__(bounds, tilt)
        #self._matrix = matrix  # type: np.ndarray
        self._inv_matrix = None  # type: Optional[np.ndarray]

    @property
    def lengths(self) -> Tuple[float, float, float]:
        """
        :return: The lengths (a, b, c) of the lattice.
        """
        return tuple(mat_lengths(self._matrix).tolist())  # type: ignore

    @property
    def angles(self) -> Tuple[float, float, float]:
        return tuple(mat_angles(m).tolist())  # type: ignore

    @property
    def is_orthogonal(self) -> bool:
        """
        :return: Whether all angles are 90 degrees.
        """
        return all(abs(a - 90) < 1e-5 for a in self.angles)

    def copy(self):
        """Deep copy of self."""
        return self.__class__(self.matrix.copy())

    @property
    def matrix(self) -> np.ndarray:
        """Copy of matrix representing the Lattice"""
        return self._matrix

    @property
    def inv_matrix(self) -> np.ndarray:
        """
        Inverse of lattice matrix.
        """
        if self._inv_matrix is None:
            self._inv_matrix = np.linalg.inv(self._matrix)
            self._inv_matrix.setflags(write=False)
        return self._inv_matrix

    def to_strings(self, significant_figures=6):
        return self.get_string(significant_figures=significant_figures)


class SeakmcData(LammpsData, MSONable):
    def __init__(
            self,
            box,
            masses,
            atoms,
            velocities=None,
            force_field=None,
            topology=None,
            atom_style="full",
            cell_dim=np.zeros(3, dtype=int),
            cusatoms=None,
            defects=None,
            def_atoms=[],
            sett=None,
    ):
        box = SeakmcBox(box.bounds, box.tilt)

        if atom_style == 'atomic':
            atom_style = 'molecular'
            im = ATOMS_HEADERS[atom_style].index('molecule-ID')
            atoms.insert(im, 'molecule-ID', np.zeros(len(atoms), dtype=int))
        elif atom_style == 'charge':
            atom_style = 'full'
            im = ATOMS_HEADERS[atom_style].index('molecule-ID')
            atoms.insert(im, 'molecule-ID', np.zeros(len(atoms), dtype=int))

        ##atoms = atoms[ATOMS_HEADERS[atom_style]]
        super().__init__(
            box,
            masses,
            atoms,
            velocities=velocities,
            force_field=force_field,
            topology=topology,
            atom_style=atom_style, )

        if self.atoms.index.has_duplicates:
            raise ValueError("The input data have duplicated atom ID.")
        self.natoms = len(self.atoms)
        try:
            self.idmax = self.atoms.index.max()
        except:
            raise ValueError(f"atom ID column is broken!")

        self.atoms = self.atoms[ATOMS_HEADERS[self.atom_style]]
        self.velocities = None

        self.cell_dim = cell_dim
        self.cell_dim_ghost = np.add(self.cell_dim, 2)
        self.cusatoms = cusatoms
        self.dangerlist = None
        self.defects = defects
        self.def_atoms = def_atoms
        if self.defects is None:
            self.ndefects = 0
        else:
            self.ndefects = len(self.defects)
        self.de_neighbors = None
        self.de_center = np.array([0.5, 0.5, 0.5])

        self.precursors = None
        self.precursor_atoms = []
        self.nprecursors = 0
        self.precursor_neighbors = None

        self.cell_dim_multiplier = None
        if sett is None:
            self.sett = Global_Variables
        else:
            self.sett = sett
        self.atoms_ghost = None
        self.natoms_ghost = 0

        self.dimension = 3
        self.PBC = [True, True, True]
        try:
            self.dimension = self.sett.data["dimension"]
        except:
            pass
        try:
            self.PBC = copy.deepcopy(self.sett.data["PBC"])
        except:
            pass

    def to_atom_style(self):
        self.atoms = self.atoms[ATOMS_HEADERS[self.atom_style]]

    def assert_settings(self, settings):
        self.sett = copy.deepcopy(settings)
        if len(self.masses) != len(self.sett.potential["masses"]):
            raise ValueError("Inconsistent number of atom types in data and input!")
        self.masses = pd.DataFrame(self.sett.potential["masses"], columns=["mass"],
                                   index=np.arange(len(self.sett.potential["masses"]), dtype=int) + 1)
        self.force_field = self.sett.potential["force_field"]

        self.dimension = self.sett.data["dimension"]
        self.PBC = copy.deepcopy(self.sett.data["PBC"])
        d = (self.sett.active_volume["DActive"] + self.sett.active_volume["DBuffer"] + self.sett.active_volume[
            "DFixed"]) * 0.6
        d = max(d, self.sett.potential["cutneighmax"] * 2.5)
        if not self.PBC[0]:
            self.box.bounds[0][0] = np.min(self.atoms["x"]) - d
            self.box.bounds[0][1] = np.max(self.atoms["x"]) + d
            self.box.tilt[0] = 0.0
            self.box.tilt[1] = 0.0
        if not self.PBC[1]:
            self.box.bounds[1][0] = np.min(self.atoms["y"]) - d
            self.box.bounds[1][1] = np.max(self.atoms["y"]) + d
            self.box.tilt[0] = 0.0
            self.box.tilt[2] = 0.0
        if not self.PBC[2]:
            self.box.bounds[2][0] = np.min(self.atoms["z"]) - d
            self.box.bounds[2][1] = np.max(self.atoms["z"]) + d
            self.box.tilt[1] = 0.0
            self.box.tilt[2] = 0.0
        self.box = SeakmcBox(self.box.bounds, self.box.tilt)

        if self.sett.active_volume["NPredef"] == 0:
            idmols = self.atoms["molecule-ID"].to_numpy().astype(int)
            idmols = np.compress(idmols < 0, idmols)
            idmols = np.unique(idmols)
            if len(idmols) > 0:
                print(f"Number of predefined defects is 0 in input.yaml!")
                print(f"But found {len(idmols)} predefined defects in data file!")
                print(f"Reset NPredef in active_volume of input.yaml and rerun the code.")
                quit()

    def insert_cusatoms(self, Sort_by='type', Ascending=True):
        cusatoms = self.atoms.copy()
        cusatoms = cusatoms.sort_values(by=Sort_by, ascending=Ascending)
        self.cusatoms = cusatoms

    def get_fractional_coords(self, df, From_Cart=True):
        if From_Cart:
            cart_coords = np.vstack((df["x"], df["y"], df["z"]))
            fractional_coords = np.dot(self.box.inv_matrix.T, cart_coords)
        else:
            fractional_coords = np.vstack((df["xsn"], df["ysn"], df["zsn"]))
        for idim in range(self.dimension):
            if self.PBC[idim]:
                fractional_coords[idim] = np.subtract(fractional_coords[idim], fractional_coords[idim].round())
                fractional_coords[idim] = np.select([fractional_coords[idim] < 0, fractional_coords[idim] >= 0],
                                                    [fractional_coords[idim] + 1.0, fractional_coords[idim]])
        dropcols = []
        if "xsn" in df.columns: dropcols.append("xsn")
        if "ysn" in df.columns: dropcols.append("ysn")
        if "zsn" in df.columns: dropcols.append("zsn")
        if len(dropcols) > 0: df = df.drop(dropcols, axis=1)
        df.insert(len(df.columns), "xsn", fractional_coords[0])
        df.insert(len(df.columns), "ysn", fractional_coords[1])
        df.insert(len(df.columns), "zsn", fractional_coords[2])
        return df

    def insert_tags(self, df):
        tags = df.index.to_numpy().astype(int)
        if "tag" in df.columns: df = df.drop(["tag"], axis=1)
        df.insert(len(df.columns), "tag", tags)
        return df

    def insert_itags(self, df):
        itags = np.arange(len(df), dtype=int)
        if "itag" in df.columns: df = df.drop(["itag"], axis=1)
        df.insert(len(df.columns), "itag", itags)
        return df

    def init_cell_dim(self, cellcut=5):
        cell_dim = np.ones(3, dtype=int)
        for m in range(3):
            mm = int(self.box.lengths[m] / cellcut)
            if mm < 1: mm = 1
            cell_dim[m] = mm

        self.cell_dim = cell_dim
        self.cell_dim_ghost = np.add(self.cell_dim, 2)
        self.cell_dim_multiplier = np.array([
            self.cell_dim_ghost[1] * self.cell_dim_ghost[2],
            self.cell_dim_ghost[2],
            1,
        ]).astype(int)

    def insert_atoms_ghost(self, cellcut=5):
        self.atoms_ghost = None
        self.natoms_ghost = 0
        self.init_cell_dim(cellcut=cellcut)
        atoms_ghost = self.atoms.to_numpy().T

        df_atoms_ghost_columns = self.atoms.columns.tolist()

        cid = [df_atoms_ghost_columns.index("xsn"), df_atoms_ghost_columns.index("ysn"),
               df_atoms_ghost_columns.index("zsn")]
        cidn = [df_atoms_ghost_columns.index("x"), df_atoms_ghost_columns.index("y"), df_atoms_ghost_columns.index("z")]

        for i in range(len(cid)):
            c = np.multiply(atoms_ghost[cid[i]], self.cell_dim[i]).astype(int)
            ghost = np.compress(c == 0, atoms_ghost, axis=1)
            ghost[cid[i], :] = ghost[cid[i], :] + 1.0
            ghost[cidn[i], :] = ghost[cidn[i], :] + self.box.matrix[0, i] + self.box.matrix[1, i] + self.box.matrix[
                2, i]
            atoms_ghost = np.append(atoms_ghost, ghost, axis=1)
            ghost = np.compress(c == self.cell_dim[i] - 1, atoms_ghost, axis=1)
            ghost[cid[i], :] = ghost[cid[i], :] - 1.0
            ghost[cidn[i], :] = ghost[cidn[i], :] - self.box.matrix[0, i] - self.box.matrix[1, i] - self.box.matrix[
                2, i]
            atoms_ghost = np.append(atoms_ghost, ghost, axis=1)

        atoms_ghost = atoms_ghost.T
        self.atoms_ghost = pd.DataFrame(atoms_ghost, columns=df_atoms_ghost_columns)
        self.natoms_ghost = len(self.atoms_ghost)

        cx = np.multiply(self.atoms_ghost["xsn"], self.cell_dim[0])
        cx = np.select([cx < 0.0, cx < self.cell_dim[0], cx >= self.cell_dim[0]],
                       [0, cx.astype(int) + 1, self.cell_dim[0] + 1])
        cy = np.multiply(self.atoms_ghost["ysn"], self.cell_dim[1])
        cy = np.select([cy < 0.0, cy < self.cell_dim[1], cy >= self.cell_dim[1]],
                       [0, cy.astype(int) + 1, self.cell_dim[1] + 1])
        cz = np.multiply(self.atoms_ghost["zsn"], self.cell_dim[2])
        cz = np.select([cz < 0.0, cz < self.cell_dim[2], cz >= self.cell_dim[2]],
                       [0, cz.astype(int) + 1, self.cell_dim[2] + 1])
        cell_coords = np.vstack((cx, cy, cz)).astype(int)
        return cell_coords

    def insert_atoms_cell(self, cellcut=5):
        cell_coords = self.insert_atoms_ghost(cellcut=cellcut)
        idc = np.dot(cell_coords.T, self.cell_dim_multiplier)
        if "idc" in self.atoms_ghost.columns: self.atoms_ghost = self.atoms_ghost.drop(["idc"], axis=1)
        self.atoms_ghost.insert(len(self.atoms_ghost.columns), "idc", idc)

        if "idc" in self.atoms.columns: self.atoms = self.atoms.drop(["idc"], axis=1)
        self.atoms.insert(len(self.atoms.columns), "idc", idc[0:self.natoms])

    def insert_df_ghost(self, df, nReal, cellcut=5):
        self.init_cell_dim(cellcut=cellcut)
        df = df.truncate(after=df.index[nReal - 1])
        atoms_ghost = df.to_numpy().T
        df_atoms_ghost_columns = df.columns.tolist()
        isCart = False
        if 'x' in df_atoms_ghost_columns:
            isCart = True
            cidn = [df_atoms_ghost_columns.index("x"), df_atoms_ghost_columns.index("y"),
                    df_atoms_ghost_columns.index("z")]
        cid = [df_atoms_ghost_columns.index("xsn"), df_atoms_ghost_columns.index("ysn"),
               df_atoms_ghost_columns.index("zsn")]
        for i in range(len(cid)):
            c = np.multiply(atoms_ghost[cid[i]], self.cell_dim[i]).astype(int)
            ghost = np.compress(c == 0, atoms_ghost, axis=1)
            ghost[cid[i], :] = ghost[cid[i], :] + 1.0
            if isCart: ghost[cidn[i], :] = (ghost[cidn[i], :] + self.box.matrix[0, i] + self.box.matrix[1, i] +
                                            self.box.matrix[2, i])
            atoms_ghost = np.append(atoms_ghost, ghost, axis=1)
            ghost = np.compress(c == self.cell_dim[i] - 1, atoms_ghost, axis=1)
            ghost[cid[i], :] = ghost[cid[i], :] - 1.0
            if isCart: ghost[cidn[i], :] = (ghost[cidn[i], :] - self.box.matrix[0, i] - self.box.matrix[1, i] -
                                            self.box.matrix[2, i])
            atoms_ghost = np.append(atoms_ghost, ghost, axis=1)

        atoms_ghost = atoms_ghost.T
        df = pd.DataFrame(atoms_ghost, columns=df_atoms_ghost_columns)
        cx = np.multiply(df["xsn"], self.cell_dim[0])
        cx = np.select([cx < 0.0, cx < self.cell_dim[0], cx >= self.cell_dim[0]],
                       [0, cx.astype(int) + 1, self.cell_dim[0] + 1])
        cy = np.multiply(df["ysn"], self.cell_dim[1])
        cy = np.select([cy < 0.0, cy < self.cell_dim[1], cy >= self.cell_dim[1]],
                       [0, cy.astype(int) + 1, self.cell_dim[1] + 1])
        cz = np.multiply(df["zsn"], self.cell_dim[2])
        cz = np.select([cz < 0.0, cz < self.cell_dim[2], cz >= self.cell_dim[2]],
                       [0, cz.astype(int) + 1, self.cell_dim[2] + 1])
        cell_coords = np.vstack((cx, cy, cz)).astype(int)
        return df, cell_coords

    def insert_df_cell(self, df, nReal, cellcut=5):
        df, cell_coords = self.insert_df_ghost(df, nReal, cellcut=cellcut)
        idc = np.dot(cell_coords.T, self.cell_dim_multiplier)
        if "idc" in df.columns: df = df.drop(["idc"], axis=1)
        df.insert(len(df.columns), "idc", idc)
        return df

    def generate_thisdtype(self, df, OutIndex=True):
        cols = df.columns.tolist()
        if OutIndex:
            thisdtype = [("index", "<i8")]
        else:
            thisdtype = []

        for i in range(len(cols)):
            key = cols[i]
            if key == "molecule-ID" or key == "type" or key == "tag" or key == "idc" or key == "itag":
                thisdtype += [(key, "<i8")]
            elif key == "q" or key == "x" or key == "y" or key == "z" or key == "xsn" or key == "ysn" or key == "zsn":
                thisdtype += [(key, "<f8")]
            elif key[0] == "id":
                thisdtype += [(key, "<i8")]
            else:
                thisdtype += [(key, "<f8")]
        return thisdtype

    def atoms_to_array(self, df, OutIndex=True):
        thisdtype = self.generate_thisdtype(df, OutIndex=OutIndex)
        atoms_array = np.array(list(df.itertuples(index=OutIndex, name=None)), dtype=thisdtype)
        return atoms_array

    def group_atoms_by(self, group_by, atoms_array):
        atom_dtype = atoms_array.dtype
        atom_entries = [key for key in atom_dtype.fields]
        if group_by not in atom_entries:
            raise ValueError(group_by + " is not in atom entries!")

        atoms_array = atoms_array[atoms_array[group_by].argsort()]
        ### info_tup: [0] idsort values, [1] indices starting from 0, [2] counts of entries
        info_tuples = np.unique(atoms_array[group_by], return_index=True, return_counts=True)
        group_lists = np.split(atoms_array, info_tuples[1][1:])

        return group_lists, info_tuples, atom_dtype

    def get_cell_atoms(self, idself, thiscoords, idc_reverse, grouped_atoms, atomdtype, Self_Excluded=False,
                       isHalf=False, indkey="itag"):
        if isinstance(atomdtype, np.dtype):
            if not indkey in atomdtype.fields:
                raise ValueError("The atoms must have itag entry!")
        else:
            raise ValueError("The atomdtype must be numpy dtype.")

        thisidcxyz = np.multiply(thiscoords, self.cell_dim).astype(int) + 1
        thisncid = np.dot(np.add(cell_neigh_array, thisidcxyz), self.cell_dim_multiplier)
        inds = np.array([], dtype=int)
        thisatoms = np.array([], dtype=atomdtype)
        for cid in np.nditer(thisncid[0:]):
            ids = idc_reverse[cid]
            if ids >= 0:
                thisatoms = np.hstack((thisatoms, grouped_atoms[ids]))
                inds = np.hstack((inds, grouped_atoms[ids][indkey]))

        if isHalf and Self_Excluded:
            thisatoms = np.compress(inds > idself, thisatoms, axis=0)
            inds = np.compress(inds > idself, inds, axis=0)
        elif isHalf:
            thisatoms = np.compress(inds >= idself, thisatoms, axis=0)
            inds = np.compress(inds >= idself, inds, axis=0)
        elif Self_Excluded:
            thisatoms = np.compress(inds != idself, thisatoms, axis=0)
            inds = np.compress(inds != idself, inds, axis=0)
        return thisatoms

    def build_neighbor_list(self, df, nReal, cutneighmax, Style="itag", isHalf=False):
        if "idc" not in df.columns:
            raise ValueError("The entry must contain cell id in build_neighbor_list!")
        if "itag" not in df.columns:
            raise ValueError("The entry must contain itag in build_neighbor_list!")

        indkey = "itag"
        if Style == "index": indkey = "index"
        if isinstance(cutneighmax, list):
            nlist = len(cutneighmax)
            cutneighmaxsq = np.array(cutneighmax) * np.array(cutneighmax)
            logstr = "Cutneighmax must be sorted descendingly."
            print(logstr)
            #raise ValueError(logstr)
        else:
            nlist = 1
            cutneighmaxsq = np.array([cutneighmax]) * np.array([cutneighmax])

        atoms_ghost_array = self.atoms_to_array(df, OutIndex=True)
        grouped_atoms, group_info, atomdtype = self.group_atoms_by("idc", atoms_ghost_array)

        ntotcell = self.cell_dim_ghost[0] * self.cell_dim_ghost[1] * self.cell_dim_ghost[2]

        idc_reverse = np.array([-1] * ntotcell, dtype=int)
        for i in range(len(group_info[0])):
            idc_reverse[group_info[0][i]] = i

        nnlist = [[] for _ in range(nlist)]
        for nr in range(nReal):
            thiscoords = np.array([df.iloc[nr]["xsn"], df.iloc[nr]["ysn"], df.iloc[nr]["zsn"]])
            thisatoms = self.get_cell_atoms(nr, thiscoords, idc_reverse, grouped_atoms, atomdtype, Self_Excluded=False,
                                            isHalf=isHalf)
            inds = np.array(thisatoms[indkey], dtype=int)
            xyzs = np.vstack((thisatoms["xsn"], thisatoms["ysn"], thisatoms["zsn"]))
            for icut in range(nlist):
                if inds.shape[0] > 0:
                    #thisxyznsq = np.sum((xyzs.T - thiscoords)**2, axis = 1)
                    thisxyznsq = np.sum(np.dot((xyzs.T - thiscoords), self.box.matrix) ** 2, axis=1)
                    inds = np.compress(thisxyznsq < cutneighmaxsq[icut], inds, axis=0)
                    thisatoms = np.compress(thisxyznsq < cutneighmaxsq[icut], thisatoms, axis=0)
                    thisxyznsq = np.compress(thisxyznsq < cutneighmaxsq[icut], thisxyznsq, axis=0)
                    if Style == "itag" or Style == "index":
                        nnlist[icut].append(inds)
                    else:
                        nnlist[icut].append(thisatoms)  ##nnlist.append(inds)
                else:
                    if Style == "itag" or Style == "index":
                        nnlist[icut].append(np.array([], dtype=int))
                    else:
                        nnlist[icut].append(np.array([], dtype=atomdtype))
        if nlist == 1:
            return nnlist[0]
        else:
            return nnlist

    def generate_defect_atom(self, id, xyz, xyzsn, thisdtype=None):
        if not isinstance(thisdtype, list): thisdtype = self.generate_thisdtype(self.atoms, OutIndex=True)
        thisatom = []
        for key in thisdtype:
            if key[0] == "index":
                thisatom.append(id)
            elif key[0] == "q":
                thisatom.append(0.0)
            elif key[0] == "xsn":
                thisatom.append(xyzsn[0])
            elif key[0] == "ysn":
                thisatom.append(xyzsn[1])
            elif key[0] == "zsn":
                thisatom.append(xyzsn[2])
            elif key[0] == "x":
                thisatom.append(xyz[0])
            elif key[0] == "y":
                thisatom.append(xyz[1])
            elif key[0] == "z":
                thisatom.append(xyz[2])
            elif key[0] == "itag":
                thisatom.append(self.natoms + id + 1)
            else:
                thisatom.append(0)
        thisatom = tuple(thisatom)
        thisatom = np.array([thisatom], dtype=thisdtype)
        return thisatom

    def realtime_set_MolID(self, last_de_center):
        if last_de_center is None:
            pass
        else:
            rcut = self.sett.active_volume["R4RT_SetMolID"]
            self.insert_atoms_cell(cellcut=rcut * 1.2)
            atoms_ghost_array = self.atoms_to_array(self.atoms_ghost, OutIndex=True)

            ntotcell = self.cell_dim_ghost[0] * self.cell_dim_ghost[1] * self.cell_dim_ghost[2]
            grouped_atoms, group_info, atomdtype = self.group_atoms_by("idc", atoms_ghost_array)

            idc_reverse = np.array([-1] * ntotcell, dtype=int)
            for i in range(len(group_info[0])):
                idc_reverse[group_info[0][i]] = i

            masks = np.zeros(self.natoms, dtype=bool)
            thisatoms = self.get_cell_atoms(0, last_de_center, idc_reverse, grouped_atoms, atomdtype,
                                            Self_Excluded=False, isHalf=False)
            xyzs = np.vstack((thisatoms["xsn"], thisatoms["ysn"], thisatoms["zsn"]))
            inds = np.array(thisatoms["itag"], dtype=int)
            if inds.shape[0] > 0:
                thisxyznsq = np.sum(np.dot((xyzs.T - last_de_center), self.box.matrix) ** 2, axis=1)
                inds = np.compress(thisxyznsq < rcut ** 2, inds, axis=0)
                masks[inds] = True

            self.atoms_ghost = None
            self.natoms_ghost = 0
            npredef = self.sett.active_volume["NPredef"]
            idmols = self.atoms["molecule-ID"].to_numpy().astype(int)
            idmols = np.select([idmols < 0, idmols >= 0], [0, idmols])
            idmols = np.select([masks == False, masks == True], [-npredef - 1, idmols])

            tlist = self.sett.active_volume["FCT4RT_SetMolID"]
            for i in range(3):
                isFCT = False
                if isinstance(tlist[i * 2], float):
                    isFCT = True
                    thismin = tlist[i * 2]
                else:
                    thismin = -1.1
                if isinstance(tlist[i * 2 + 1], float):
                    isFCT = True
                    thismax = tlist[i * 2 + 1]
                else:
                    thismax = 1.1

                if isFCT:
                    if i == 0:
                        x = self.atoms["xsn"].to_numpy()
                    elif i == 1:
                        x = self.atoms["ysn"].to_numpy()
                    elif i == 2:
                        x = self.atoms["zsn"].to_numpy()
                    idmols = np.select([x < thismin, x < thismax, x >= thismax], [-npredef - 1, idmols, -npredef - 1])

            self.atoms = self.atoms.drop(["molecule-ID"], axis=1)
            im = ATOMS_HEADERS[self.atom_style].index('molecule-ID')
            self.atoms.insert(im, 'molecule-ID', idmols)

    def BLCN_find_defects(self):
        cutdefectmax = self.sett.active_volume['cutdefectmax']
        cutdefectmaxsq = cutdefectmax * cutdefectmax
        method = self.sett.active_volume['FindDefects']['Method'].upper()

        if self.sett.active_volume["NPredef"] > 0:
            idmols = self.atoms["molecule-ID"].to_numpy().astype(int)
            ind0s = self.atoms["itag"].to_numpy().astype(int)
            ind0s = np.compress(idmols >= 0, ind0s, axis=0)
        else:
            ind0s = np.arange(self.natoms, dtype=int)

        atoms_ghost_array = self.atoms_to_array(self.atoms_ghost, OutIndex=True)

        ntotcell = self.cell_dim_ghost[0] * self.cell_dim_ghost[1] * self.cell_dim_ghost[2]
        grouped_atoms, group_info, atomdtype = self.group_atoms_by("idc", atoms_ghost_array)

        DCut4Def = self.sett.active_volume['FindDefects']['DCut4Def']
        idc_reverse = np.array([-1] * ntotcell, dtype=int)
        for i in range(len(group_info[0])):
            idc_reverse[group_info[0][i]] = i

        defect_list = []
        dCN_list = []
        for inr in range(len(ind0s)):
            nr = ind0s[inr]
            isdefect = False
            defecttype = None
            thisidmol = self.atoms.iloc[nr]["molecule-ID"]
            thisdCN = 0
            ismolid = True
            if thisidmol < 0:
                ismolid = False
            else:
                if isinstance(self.sett.active_volume["FindDefects"]["MolIDCap"], int):
                    if thisidmol >= self.sett.active_volume["FindDefects"]["MolIDCap"]:
                        ismolid = False
            if ismolid:
                thiscoords = np.array(
                    [self.atoms.iloc[nr]["xsn"], self.atoms.iloc[nr]["ysn"], self.atoms.iloc[nr]["zsn"]])
                thistype = int(self.atoms.iloc[nr]["type"]) - 1

                if 'Q' in method:
                    qtype = self.sett.potential['charges'][thistype]
                    try:
                        thisq = self.atoms.iloc[nr]["q"]
                        if abs(thisq - qtype) >= self.sett.potential['qtolerances'][thistype]: isdefect = True
                    except:
                        pass

                if not isdefect:
                    if 'CN' in method or 'BL' in method:
                        thisatoms = self.get_cell_atoms(nr, thiscoords, idc_reverse, grouped_atoms, atomdtype,
                                                        Self_Excluded=True, isHalf=False)
                        inds = np.array(thisatoms["itag"], dtype=int)
                        xyzs = np.vstack((thisatoms["xsn"], thisatoms["ysn"], thisatoms["zsn"]))
                        types = np.array(thisatoms["type"], dtype=int)

                        if inds.shape[0] <= 0:
                            isdefect = True
                        else:
                            #thisxyznsq = np.sum((xyzs.T - thiscoords)**2, axis = 1)
                            thisxyznsq = np.sum(np.dot((xyzs.T - thiscoords), self.box.matrix) ** 2, axis=1)
                            inds = np.compress(thisxyznsq < cutdefectmaxsq, inds, axis=0)
                            types = np.compress(thisxyznsq < cutdefectmaxsq, types, axis=0)
                            thisxyznsq = np.compress(thisxyznsq < cutdefectmaxsq, thisxyznsq, axis=0)
                            types = types - 1
                            masks = np.zeros(inds.shape, dtype=bool)
                            for j in range(inds.shape[0]):
                                thiscut = self.sett.potential['cutneighs'][thistype][types[j]]
                                thiscutsq = thiscut * thiscut
                                if thisxyznsq[j] < thiscutsq: masks[j] = True

                            thisxyznsq = thisxyznsq[masks]
                            types = types[masks]
                            inds = inds[masks]

                            cntype = self.sett.potential['coordnums'][thistype]
                            thisdCN = inds.shape[0] - cntype
                            if self.sett.active_volume["FindDefects"]["DiscardType"][
                               0:2].upper() == "UN" and thisdCN < 0:
                                isdefect = True
                            elif self.sett.active_volume["FindDefects"]["DiscardType"][
                                 0:2].upper() == "OV" and thisdCN > 0:
                                isdefect = True
                            if not isdefect:
                                if 'CN' in method:
                                    if inds.shape[0] - cntype > 0.5:
                                        isdefect = True
                                    elif inds.shape[0] - cntype < -0.5:
                                        isdefect = True

                                    if not isdefect:
                                        if 'BL' in method:
                                            thisscale = (1.0 - 0.2 * (cntype - inds.shape[0]) / cntype)
                                            thisscale = min(thisscale, 1.2)
                                            thisscale = max(thisscale, 0.8)
                                            for j in range(inds.shape[0]):
                                                bl = self.sett.potential['bondlengths'][thistype][types[j]] * thisscale
                                                if thisxyznsq[j] < bl * bl * (1.0 - DCut4Def) * (1.0 - DCut4Def):
                                                    isdefect = True
                                                    break
                                                if thisxyznsq[j] > bl * bl * (1.0 + DCut4Def) * (1.0 + DCut4Def):
                                                    isdefect = True
                                                    break
                                else:
                                    thisscale = 1.0
                                    for j in range(inds.shape[0]):
                                        bl = self.sett.potential['bondlengths'][thistype][types[j]] * thisscale
                                        if thisxyznsq[j] < bl * bl * (1.0 - DCut4Def) * (1.0 - DCut4Def):
                                            isdefect = True
                                            break
                                        if thisxyznsq[j] > bl * bl * (1.0 + DCut4Def) * (1.0 + DCut4Def):
                                            isdefect = True
                                            break
            if isdefect:
                defect_list.append(atoms_ghost_array[nr])
                dCN_list.append(thisdCN)

        return defect_list, dCN_list

    def WS_find_defects(self):
        cutdefectmax = self.sett.active_volume['cutdefectmax']
        DCut4Def = self.sett.active_volume['FindDefects']['DCut4Def'] * cutdefectmax
        DCut4Defsq = DCut4Def * DCut4Def

        atoms_ghost_array = self.atoms_to_array(self.atoms_ghost, OutIndex=True)
        ntotcell = self.cell_dim_ghost[0] * self.cell_dim_ghost[1] * self.cell_dim_ghost[2]
        grouped_atoms, group_info, atomdtype = self.group_atoms_by("idc", atoms_ghost_array)

        refdata = SeakmcData.from_file(self.sett.active_volume['FindDefects']["ReferenceData"],
                                       atom_style=self.sett.active_volume['FindDefects']["atom_style4Ref"])
        refdata.to_atom_style()
        if mat_mag(refdata.box.matrix - self.box.matrix) >= 1.0e-8:
            raise ValueError("Reference data has a different lattice!")
        refdata.atoms = refdata.get_fractional_coords(refdata.atoms)
        refdata.atoms = refdata.insert_tags(refdata.atoms)
        refdata.atoms = refdata.insert_itags(refdata.atoms)
        refdata.insert_atoms_cell(cellcut=cutdefectmax * 1.2)
        ref_atoms_ghost_array = refdata.atoms_to_array(refdata.atoms_ghost, OutIndex=True)
        ref_groups, ref_info, ref_dtype = refdata.group_atoms_by("idc", ref_atoms_ghost_array)

        idc_reverse = np.array([-1] * ntotcell, dtype=int)
        for i in range(len(ref_info[0])):
            idc_reverse[ref_info[0][i]] = i

        tv = np.zeros(refdata.natoms, dtype=int)
        ti = np.zeros(self.natoms, dtype=int)
        for nr in range(refdata.natoms):
            thisidmol = refdata.atoms.iloc[nr]["molecule-ID"]
            ismolid = True
            if thisidmol < 0:
                ismolid = False
            else:
                if isinstance(self.sett.active_volume["FindDefects"]["MolIDCap"], int):
                    if thisidmol >= self.sett.active_volume["FindDefects"]["MolIDCap"]:
                        ismolid = False
            if ismolid:
                thiscoords = np.array(
                    [refdata.atoms.iloc[nr]["xsn"], refdata.atoms.iloc[nr]["ysn"], refdata.atoms.iloc[nr]["zsn"]])
                thisatoms = self.get_cell_atoms(nr, thiscoords, idc_reverse, grouped_atoms, atomdtype,
                                                Self_Excluded=False, isHalf=False)
                xyzs = np.vstack((thisatoms["xsn"], thisatoms["ysn"], thisatoms["zsn"]))
                inds = np.array(thisatoms["itag"], dtype=int)
                if inds.shape[0] > 0:
                    thisxyznsq = np.sum(np.dot((xyzs.T - thiscoords), refdata.box.matrix) ** 2, axis=1)
                    inds = np.compress(thisxyznsq < DCut4Defsq, inds, axis=0)

                    tv[nr] = inds.shape[0]
                    for j in range(inds.shape[0]):
                        jj = inds[j]
                        ti[jj] = nr

        defect_list = []
        dCN_list = []
        if self.sett.active_volume["FindDefects"]["DiscardType"][0:2].upper() != "UN":
            indv = np.where(tv == 0)
            for i in range(indv[0].shape[0]):
                j = indv[0][i]
                defect_list.append(ref_atoms_ghost_array[j])
                dCN_list.append(-1)

        if self.sett.active_volume["FindDefects"]["DiscardType"][0:2].upper() != "OV":
            indi = np.where(ti == 0)
            for i in range(indi[0].shape[0]):
                j = indi[0][i]
                defect_list.append(ref_atoms_ghost_array[j])
                dCN_list.append(1)
        return defect_list, dCN_list

    def custom_find_defects(self):
        defects = self.sett.active_volume['FindDefects']['Defects']
        thisdtype = self.generate_thisdtype(self.atoms, OutIndex=True)
        defect_list = np.array([], dtype=thisdtype)
        dCN_list = []
        for i in range(len(defects)):
            xyz = np.array([defects[i][0], defects[i][1], defects[i][2]])
            center = np.dot(xyz, self.box.inv_matrix)
            thisatom = self.generate_defect_atom(i, xyz, center, thisdtype=thisdtype)
            defect_list = np.hstack((defect_list, thisatom))
            dCN_list.append(0)
        return defect_list, dCN_list

    def find_defects(self):
        av_sett = self.sett.active_volume
        npredef = av_sett["NPredef"]
        Find_Defects = True
        thisdtype = self.generate_thisdtype(self.atoms, OutIndex=True)
        defect_list = np.array([], dtype=thisdtype)
        if npredef > 0:
            atoms_array = self.atoms_to_array(self.atoms, OutIndex=True)
            def_groups, def_info, atomdtype = self.group_atoms_by("molecule-ID", atoms_array)
            if def_info[0].shape[0] <= npredef: Find_Defects = False
            for i in range(min(npredef, def_info[0].shape[0])):
                if def_info[0][i] < -npredef:
                    pass
                elif def_info[0][i] < 0:
                    defect_list = np.hstack((defect_list, def_groups[i]))
                else:
                    Find_Defects = True
        if len(defect_list) > 0:
            dCN_list = [0] * len(defect_list)
        else:
            dCN_list = []

        if Find_Defects and av_sett["PredefOnly"] == False:
            method = av_sett['FindDefects']['Method'].upper()
            if "WS" in method:
                defect_list2, dCN_list2 = self.WS_find_defects()
            elif method[0:3].upper() == "CUS":
                defect_list2, dCN_list2 = self.custom_find_defects()
            elif "CN" in method or "BL" in method or "Q" in method:
                defect_list2, dCN_list2 = self.BLCN_find_defects()
            else:
                logstr = f"The {method} for finding defects is not Valid."
                print(logstr)
                quit()
        else:
            defect_list2 = np.array([], dtype=thisdtype)
            dCN_list2 = []

        if len(defect_list2) > 0:
            defect_list = np.hstack((defect_list, defect_list2))
            dCN_list += dCN_list2
        return defect_list, dCN_list

    def get_defects_from_input(self, nactive):
        if "xsn" not in self.atoms.columns: self.atoms = self.get_fractional_coords(self.atoms)
        thisdtype = self.generate_thisdtype(self.atoms, OutIndex=True)
        defect_list = np.array([], dtype=thisdtype)

        xsns = self.atoms["xsn"][0:nactive]
        ysns = self.atoms["ysn"][0:nactive]
        zsns = self.atoms["zsn"][0:nactive]
        center = np.array([xsns.mean(), ysns.mean(), zsns.mean()])
        xyz = np.dot(self.box.matrix.T, center)
        thisatom = self.generate_defect_atom(0, xyz, center, thisdtype=thisdtype)
        defect_list = np.append(defect_list, thisatom)
        if len(defect_list) > 0:
            dCN_list = [0] * len(defect_list)
        else:
            dCN_list = []
        return defect_list, dCN_list

    def find_df_chains(self, df, rcut, nReal, Recursive=False, Overlapping=True, Order4Recursive=None,
                       Overlap4OrderRecursive=True):
        rcutsq = rcut * rcut
        thisdtype = self.generate_thisdtype(df, OutIndex=True)
        atoms_ghost_array = self.atoms_to_array(df, OutIndex=True)
        grouped_atoms, group_info, atomdtype = self.group_atoms_by("idc", atoms_ghost_array)
        ntotcell = self.cell_dim_ghost[0] * self.cell_dim_ghost[1] * self.cell_dim_ghost[2]
        idc_reverse = np.array([-1] * ntotcell, dtype=int)
        for i in range(len(group_info[0])):
            idc_reverse[group_info[0][i]] = i

        #########################################################################################################
        def update_chain_info(nReal, nr, thiscoords, id_chain, indchain, nchain, arrays, masks, next_id, id_left,
                              array_atoms, orders,
                              Recursive=False, Overlapping=True, Order4Recursive=None, Overlap4OrderRecursive=True):
            def update_ichain_info(nr, ichain, thiscenter, indchain, inds, id_chain, arrays):
                icen = np.array([arrays[ichain]["xsn"], arrays[ichain]["ysn"], arrays[ichain]["zsn"]])
                jcen = np.array([thiscenter["xsn"], thiscenter["ysn"], thiscenter["zsn"]])
                cshift = icen - jcen
                for idim in range(self.dimension):
                    if self.PBC[idim]:
                        if cshift[idim] < -0.5:
                            cshift[idim] = -1.0
                        elif cshift[idim] < 0.5:
                            cshift[idim] = 0.0
                        else:
                            cshift[idim] = 1.0
                icen = (icen * indchain[ichain].shape[0] + (jcen + cshift) * inds.shape[0]) / (
                            indchain[ichain].shape[0] + inds.shape[0])
                arrays[ichain]["xsn"] = icen[0]
                arrays[ichain]["ysn"] = icen[1]
                arrays[ichain]["zsn"] = icen[2]
                idCN = arrays[ichain]["dCN"]
                jdCN = thiscenter["dCN"]
                idCN = (idCN * indchain[ichain].shape[0] + jdCN * inds.shape[0]) / (
                            indchain[ichain].shape[0] + inds.shape[0])
                arrays[ichain]["dCN"] = idCN
                id_chain[inds] = ichain
                indchain[ichain] = np.hstack((indchain[ichain], inds))
                return indchain, id_chain

            def update_id_left(inds, next_id, id_left, masks, orders, iorder, nr, IntersectOnly=False):
                ij, i_ind, j_ind = np.intersect1d(inds, id_left, return_indices=True)
                if IntersectOnly:
                    thisij = np.copy(ij)
                else:
                    thisij = np.copy(inds)
                if nr in thisij:
                    thisij = np.hstack((np.array([nr]), thisij[thisij != nr]))
                next_id = np.hstack((next_id, thisij))
                id_left = np.delete(id_left, j_ind)
                masks[inds] = False
                orders[inds] = iorder + 1
                thisij = None
                return next_id, id_left, masks, orders

            ichain = id_chain[nr]
            iorder = orders[nr]
            thisatoms = self.get_cell_atoms(nr, thiscoords, idc_reverse, grouped_atoms, atomdtype, Self_Excluded=False,
                                            isHalf=False)
            inds = np.array(thisatoms["itag"], dtype=int)
            xyzs = np.vstack((thisatoms["xsn"], thisatoms["ysn"], thisatoms["zsn"]))
            thisxyznsq = np.sum(np.dot((xyzs.T - thiscoords), self.box.matrix) ** 2, axis=1)
            inds = np.compress(thisxyznsq < rcutsq, inds, axis=0)
            thisatoms = np.compress(thisxyznsq < rcutsq, thisatoms, axis=0)

            DoRecursive = True
            if Recursive and isinstance(Order4Recursive, int):
                if iorder < Order4Recursive:
                    DoRecursive = True
                else:
                    DoRecursive = False
                thisorders = orders[inds]
                if Overlap4OrderRecursive:
                    inds = np.compress(thisorders <= Order4Recursive, inds, axis=0)
                    thisatoms = np.compress(thisorders <= Order4Recursive, thisatoms, axis=0)
                else:
                    inds = np.compress(thisorders < Order4Recursive, inds, axis=0)
                    thisatoms = np.compress(thisorders < Order4Recursive, thisatoms, axis=0)

            if masks[nr]:
                thismask = np.zeros(inds.shape[0], dtype=bool)
                thismask[masks[inds]] = True
                indsF = inds[np.invert(thismask)]
                if inds.shape[0] > 0:
                    if indsF.shape[0] > 0 and Recursive:
                        tmporder = orders[indsF[0]]
                        MakeNew = False
                        if tmporder < Order4Recursive:
                            MakeNew = False
                        else:
                            MakeNew = True
                        if MakeNew:
                            #print("MASK is true and indsF>0 and make a NEW chain")
                            if not Overlap4OrderRecursive:
                                inds = inds[thismask]
                                thisatoms = thisatoms[thismask]
                            if not Overlapping:
                                array_atoms.append(thisatoms)
                            center = copy.deepcopy(thisatoms[0])
                            center["xsn"] = np.mean(np.array(thisatoms["xsn"], dtype=float))
                            center["ysn"] = np.mean(np.array(thisatoms["ysn"], dtype=float))
                            center["zsn"] = np.mean(np.array(thisatoms["zsn"], dtype=float))
                            center["dCN"] = np.mean(np.array(thisatoms["dCN"], dtype=float))
                            arrays = np.hstack((arrays, np.array(center, dtype=thisdtype)))
                            ichain = nchain
                            id_chain[inds] = ichain
                            indchain.append(inds)

                            masks = np.ones(nReal, dtype=bool)
                            #orders=np.zeros(nReal, dtype=int)
                            if Overlap4OrderRecursive:
                                next_id, id_left, masks, orders = update_id_left(inds, next_id, id_left, masks, orders,
                                                                                 0, nr, IntersectOnly=False)
                            else:
                                next_id, id_left, masks, orders = update_id_left(inds, next_id, id_left, masks, orders,
                                                                                 0, nr, IntersectOnly=True)
                            nchain += 1
                        else:
                            #print("MASK is true and indsF>0 and NOT make a new chain")
                            ichain = id_chain[indsF[0]]
                            iorder = orders[indsF[0]]
                            inds = inds[thismask]
                            thisatoms = thisatoms[thismask]
                            if not Overlapping:
                                array_atoms[ichain] = np.hstack((array_atoms[ichain], thisatoms))
                            center = copy.deepcopy(thisatoms[0])
                            center["xsn"] = np.mean(np.array(thisatoms["xsn"], dtype=float))
                            center["ysn"] = np.mean(np.array(thisatoms["ysn"], dtype=float))
                            center["zsn"] = np.mean(np.array(thisatoms["zsn"], dtype=float))
                            center["dCN"] = np.mean(np.array(thisatoms["dCN"], dtype=float))
                            indchain, id_chain = update_ichain_info(nr, ichain, center, indchain, inds, id_chain,
                                                                    arrays)
                            next_id, id_left, masks, orders = update_id_left(inds, next_id, id_left, masks, orders,
                                                                             iorder, nr, IntersectOnly=True)
                    else:
                        #if Recursive: print("MASK is TRUE a new chain")
                        if not Overlapping:
                            array_atoms.append(thisatoms)
                        center = copy.deepcopy(thisatoms[0])
                        center["xsn"] = np.mean(np.array(thisatoms["xsn"], dtype=float))
                        center["ysn"] = np.mean(np.array(thisatoms["ysn"], dtype=float))
                        center["zsn"] = np.mean(np.array(thisatoms["zsn"], dtype=float))
                        center["dCN"] = np.mean(np.array(thisatoms["dCN"], dtype=float))
                        arrays = np.hstack((arrays, np.array(center, dtype=thisdtype)))
                        ichain = nchain
                        id_chain[inds] = ichain
                        indchain.append(inds)

                        masks = np.ones(nReal, dtype=bool)
                        #orders=np.zeros(nReal, dtype=int)
                        if Overlap4OrderRecursive:
                            next_id, id_left, masks, orders = update_id_left(inds, next_id, id_left, masks, orders, 0,
                                                                             nr, IntersectOnly=False)
                        else:
                            next_id, id_left, masks, orders = update_id_left(inds, next_id, id_left, masks, orders, 0,
                                                                             nr, IntersectOnly=True)
                        nchain += 1
            else:
                if DoRecursive:
                    thismask = np.ones(inds.shape[0], dtype=bool)
                    ij, i_ind, j_ind = np.intersect1d(indchain[ichain], inds, return_indices=True)
                    thismask[j_ind] = False
                    indsF = inds[np.invert(thismask)]
                    inds = inds[thismask]
                    thisatoms = thisatoms[thismask]
                    if inds.shape[0] > 0:
                        if not Overlapping:
                            array_atoms[ichain] = np.hstack((array_atoms[ichain], thisatoms))
                        center = copy.deepcopy(thisatoms[0])
                        center["xsn"] = np.mean(np.array(thisatoms["xsn"], dtype=float))
                        center["ysn"] = np.mean(np.array(thisatoms["ysn"], dtype=float))
                        center["zsn"] = np.mean(np.array(thisatoms["zsn"], dtype=float))
                        center["dCN"] = np.mean(np.array(thisatoms["dCN"], dtype=float))
                        indchain, id_chain = update_ichain_info(nr, ichain, center, indchain, inds, id_chain, arrays)
                        if Overlap4OrderRecursive:
                            next_id, id_left, masks, orders = update_id_left(inds, next_id, id_left, masks, orders,
                                                                             iorder, nr, IntersectOnly=False)
                        else:
                            next_id, id_left, masks, orders = update_id_left(inds, next_id, id_left, masks, orders,
                                                                             iorder, nr, IntersectOnly=True)
            return id_chain, indchain, nchain, arrays, masks, orders, next_id, id_left, array_atoms

        #########################################################################################################
        id_chain = np.zeros(nReal, dtype=int) - 1
        id_left = np.arange(nReal, dtype=int)
        next_id = np.array([], dtype=int)
        indchain = []
        nchain = 0
        arrays = np.array([], dtype=thisdtype)
        array_atoms = []
        masks = np.ones(nReal, dtype=bool)
        orders = np.zeros(nReal, dtype=int)
        ithis = 0
        idef = 0
        while True:
            if len(id_left) == 0: break
            try:
                nr = next_id[idef]
            except:
                nr = id_left[0]
            if Recursive:
                #print(f"before nr:{nr} mask: {masks[nr]} order:{orders[nr]}")
                #print(f"masks:{masks} orders:{orders}")
                #print(f"idef:{idef} next_id:{next_id}")
                #print(f"id_left:{id_left}")
                #print("------")
                thiscoords = np.array([df.iloc[nr]["xsn"], df.iloc[nr]["ysn"], df.iloc[nr]["zsn"]])
                id_chain, indchain, nchain, arrays, masks, orders, next_id, id_left, array_atoms = update_chain_info(
                    nReal, nr, thiscoords, id_chain, indchain, nchain,
                    arrays, masks, next_id, id_left, array_atoms, orders,
                    Recursive=Recursive, Overlapping=Overlapping, Order4Recursive=Order4Recursive,
                    Overlap4OrderRecursive=Overlap4OrderRecursive)
                #print(f"before nr:{nr} mask: {masks[nr]} order:{orders[nr]}")
                #print(f"masks:{masks} orders:{orders}")
                #print(f"idef:{idef} next_id:{next_id}")
                #print(f"id_left:{id_left}")
                #print("======")
            else:
                if masks[nr]:
                    thiscoords = np.array([df.iloc[nr]["xsn"], df.iloc[nr]["ysn"], df.iloc[nr]["zsn"]])
                    id_chain, indchain, nchain, arrays, masks, orders, next_id, id_left, array_atoms = (
                        update_chain_info(nReal, nr, thiscoords, id_chain, indchain, nchain,
                                          arrays, masks, next_id, id_left, array_atoms, orders,
                                          Recursive=Recursive, Overlapping=Overlapping, Order4Recursive=Order4Recursive,
                                          Overlap4OrderRecursive=Overlap4OrderRecursive))
                else:
                    pass
            idef += 1

        cols = df.columns.tolist()
        df = pd.DataFrame(arrays, columns=cols)
        fractional_coords = np.vstack((df["xsn"], df["ysn"], df["zsn"]))
        fractional_coords = np.subtract(fractional_coords, fractional_coords.round())
        fractional_coords = np.select([fractional_coords < 0, fractional_coords < 1.0, fractional_coords >= 1.0],
                                      [fractional_coords + 1.0, fractional_coords, fractional_coords - 1.0])
        dropcols = []
        if "xsn" in df.columns: dropcols.append("xsn")
        if "ysn" in df.columns: dropcols.append("ysn")
        if "zsn" in df.columns: dropcols.append("zsn")
        if len(dropcols) > 0: df = df.drop(dropcols, axis=1)
        df.insert(cols.index("xsn"), "xsn", fractional_coords[0])
        df.insert(cols.index("ysn"), "ysn", fractional_coords[1])
        df.insert(cols.index("zsn"), "zsn", fractional_coords[2])
        #print(df)
        if Overlapping:
            array_atoms = [np.array([]) for _ in range(len(df))]
        else:
            for i in range(len(df)):
                itags = array_atoms[i]["itag"]
                uniqueitags, inds = np.unique(itags, return_index=True)
                array_atoms[i] = array_atoms[i][inds]
                #print(f"i:{i} array:{array_atoms[i]['itag']}")
        return df, array_atoms

    def sort_defects4PDreduction(self):
        xsn = self.defects["xsn"].to_numpy() - np.mean(self.defects["xsn"].to_numpy())
        ysn = self.defects["ysn"].to_numpy() - np.mean(self.defects["ysn"].to_numpy())
        zsn = self.defects["zsn"].to_numpy() - np.mean(self.defects["zsn"].to_numpy())
        xsn = xsn - xsn.round()
        ysn = ysn - ysn.round()
        zsn = zsn - zsn.round()
        dsq = xsn ** 2 + ysn ** 2 + zsn ** 2
        self.defects["dsq"] = dsq
        self.defects = self.defects.sort_values(["dsq"], ascending=True)
        self.defects = self.defects.drop(["dsq"], axis=1)
        self.defects = self.defects.set_index(np.arange(self.ndefects, dtype=int))

    def get_defects(self, LogWriter, last_de_center=None):
        defect_cols = ["xsn", "ysn", "zsn", "idc", "dCN"]
        DActive = self.sett.active_volume["DActive"]
        cutdefectmax = self.sett.active_volume["cutdefectmax"]
        if self.sett.active_volume['Style'].upper() == 'ALL':
            defect_list, dCN_list = self.get_defects_from_input(self.natoms)
            self.defects = pd.DataFrame(defect_list, columns=self.atoms.columns)
            self.defects["dCN"] = dCN_list
        elif self.sett.active_volume['Style'].upper() == 'CUSTOM':
            defect_list, dCN_list = self.get_defects_from_input(self.sett.active_volume['NActive'])
            self.defects = pd.DataFrame(defect_list.T, columns=self.atoms.columns)
            self.defects["dCN"] = dCN_list
        else:
            self.atoms = self.get_fractional_coords(self.atoms)
            self.atoms = self.insert_tags(self.atoms)
            self.atoms = self.insert_itags(self.atoms)
            if self.sett.active_volume["RT_SetMolID"]: self.realtime_set_MolID(last_de_center)
            self.insert_atoms_cell(cellcut=cutdefectmax * 1.2)
            if self.sett.active_volume["FindDefects"]["Method"].upper() == "ALL":
                self.defects = self.atoms.copy(deep=True)
                self.defects = self.defects.reset_index(drop=True, inplace=False)
                dCN_list = [0] * len(self.defects)
                self.defects["dCN"] = dCN_list
            else:
                defect_list, dCN_list = self.find_defects()
                if len(defect_list) <= 0:
                    raise ValueError("No defect has been found!")
                self.defects = pd.DataFrame(defect_list.T, columns=self.atoms.columns)
                self.defects["dCN"] = dCN_list

        self.ndefects = len(self.defects)
        if self.ndefects < 1:
            raise ValueError("No defect has been found!")

        if self.sett.active_volume['PDReduction']:
            logstr = f"There are {self.ndefects} defects before the point defect reduction."
            LogWriter.write_data(logstr)
            if self.sett.active_volume['SortD4PDR']: self.sort_defects4PDreduction()
            DCut4PDR = self.sett.active_volume['DCut4PDR']
            DCut4PDR = min(DCut4PDR, DActive)
            self.defects = self.get_fractional_coords(self.defects, From_Cart=False)
            self.defects = self.insert_itags(self.defects)
            self.defects = self.insert_df_cell(self.defects, self.ndefects, cellcut=DCut4PDR * 1.2)
            self.defects, self.def_atoms = self.find_df_chains(self.defects, DCut4PDR, self.ndefects,
                                                               Recursive=self.sett.active_volume['RecursiveRed'],
                                                               Overlapping=True,
                                                               Order4Recursive=self.sett.active_volume[
                                                                   'Order4Recursive4PDR'], Overlap4OrderRecursive=True)
            self.ndefects = len(self.defects)
        else:
            self.def_atoms = [np.array([]) for _ in range(self.ndefects)]

        if not self.sett.active_volume["Overlapping"]:
            logstr = f"There are {self.ndefects} defects after the point defect reduction."
            LogWriter.write_data(logstr)
            if isinstance(self.sett.active_volume["DCut4noOverlap"], float):
                thiscut = self.sett.active_volume["DCut4noOverlap"]
            else:
                thiscut = (DActive + self.sett.active_volume["DBuffer"]) * 2.0
            if self.sett.active_volume["Stack4noOverlap"]: df_defects = self.defects.copy()
            self.defects = self.get_fractional_coords(self.defects, From_Cart=False)
            self.defects = self.insert_itags(self.defects)
            self.defects = self.insert_df_cell(self.defects, self.ndefects, cellcut=thiscut * 1.2)
            if self.sett.active_volume["Stack4noOverlap"]:
                defects, def_atoms_array = self.find_df_chains(self.defects, thiscut, self.ndefects, Recursive=True,
                                                               Overlapping=False,
                                                               Order4Recursive=self.sett.active_volume[
                                                                   'Order4Recursive4PDR'],
                                                               Overlap4OrderRecursive=self.sett.active_volume[
                                                                   'Overlap4OrderRecursive'])
                self.defects = pd.concat([defects, df_defects], ignore_index=True)
                self.def_atoms = def_atoms_array + self.def_atoms
            else:
                self.defects, self.def_atoms = self.find_df_chains(self.defects, thiscut, self.ndefects, Recursive=True,
                                                                   Overlapping=False,
                                                                   Order4Recursive=self.sett.active_volume[
                                                                       'Order4Recursive4PDR'],
                                                                   Overlap4OrderRecursive=self.sett.active_volume[
                                                                       'Overlap4OrderRecursive'])
            self.ndefects = len(self.defects)

        if not isinstance(self.sett.active_volume['NMax4Def'], bool):
            if self.ndefects > self.sett.active_volume['NMax4Def']:
                method = self.sett.active_volume['FindDefects']['Method']
                logstr = "The number of defects is larger than the number of defects allowed."
                logstr += "\n" + f"n defects: {self.ndefects}  ndmax: {self.sett.active_volume['NMax4Def']}"
                logstr += "\n" + f"The method of find_defects is {method}."
                if method == "WS":
                    logstr += ("\n" +
                               f"The distance cut (DCut4Def) is "
                               f"{100 * self.sett.active_volume['FindDefects']['DCut4Def']}"
                               f" % of max of cutdefect distance.")
                elif "CN" in method or "BL" in self.sett.active_volume['FindDefects']['Method']:
                    logstr += "\n" + f"The coordination numbers (coordnums) is {self.sett.potential['coordnums']}."
                    logstr += "\n" + (f"The distance cut (DCut4Def) for defects is "
                                      f"{100 * self.sett.active_volume['FindDefects']['DCut4Def']} % of bondlengths.")
                    logstr += "\n" + "Check the settings above, change them accordingly and rerun the code."
                raise ValueError(logstr)

        thiscut = DActive
        if not self.sett.active_volume["Overlapping"]:
            if isinstance(self.sett.active_volume["DCut4noOverlap"], float):
                thiscut = self.sett.active_volume["DCut4noOverlap"]
            else:
                thiscut = (DActive + self.sett.active_volume["DBuffer"]) * 2.0
        self.defects = self.get_fractional_coords(self.defects, From_Cart=False)
        self.defects = self.insert_itags(self.defects)
        self.defects = self.insert_df_cell(self.defects, self.ndefects, cellcut=thiscut * 1.2)
        self.de_neighbors = self.build_neighbor_list(self.defects, self.ndefects, thiscut, Style="itag", isHalf=False)
        self.defects = self.defects[defect_cols]
        self.defects = self.defects.truncate(after=self.defects.index[self.ndefects - 1])
        self.de_center = np.array([np.mean(self.defects["xsn"].to_numpy()), np.mean(self.defects["ysn"].to_numpy()),
                                   np.mean(self.defects["zsn"].to_numpy())])

    def df_center_atoms(self, df_atoms, Center=[0.0, 0.0, 0.0], Style='Fractional'):
        if Style[0:3].upper() == "CAR": Center = np.dot(Center, self.box.inv_matrix)
        xsn = df_atoms["xsn"].to_numpy() - Center[0]
        ysn = df_atoms["ysn"].to_numpy() - Center[1]
        zsn = df_atoms["zsn"].to_numpy() - Center[2]
        xsn = xsn - xsn.round()
        ysn = ysn - ysn.round()
        zsn = zsn - zsn.round()
        fcs = np.vstack((xsn, ysn, zsn))
        cs = np.dot(self.box.matrix.T, fcs)
        df_atoms["xsn"] = xsn
        df_atoms["ysn"] = ysn
        df_atoms["zsn"] = zsn
        df_atoms["x"] = cs[0]
        df_atoms["y"] = cs[1]
        df_atoms["z"] = cs[2]
        return df_atoms

    def df_av_sort(self, df, na, nb, nf):
        def df_sort(df, sort_by=["D", "X", "Y", "Z"], sorting_spacer=[0.3, 0.3, 0.3], sorting_shift=[0.0, 0.0, 0.0]):
            X = np.around((df["x"] + sorting_shift[0]) / sorting_spacer[0], decimals=0)
            Y = np.around((df["y"] + sorting_shift[1]) / sorting_spacer[1], decimals=0)
            Z = np.around((df["z"] + sorting_shift[2]) / sorting_spacer[2], decimals=0)
            df["X"] = X
            df["Y"] = Y
            df["Z"] = Z
            if "D" in sort_by:
                dsq = X ** 2 + Y ** 2 + Z ** 2
                df["D"] = dsq
            if "DXY" in sort_by:
                dsq = X ** 2 + Y ** 2
                df["DXY"] = dsq
            if "DXZ" in sort_by:
                dsq = X ** 2 + Z ** 2
                df["DXZ"] = dsq
            if "DYZ" in sort_by:
                dsq = Y ** 2 + Z ** 2
                df["DYZ"] = dsq
            if "DX" in sort_by:
                df["DX"] = np.absolute(X)
            if "DY" in sort_by:
                df["DY"] = np.absolute(Y)
            if "DZ" in sort_by:
                df["DZ"] = np.absolute(Z)

            df = df.sort_values(sort_by, ascending=True)
            df = df.drop(["X"], axis=1)
            df = df.drop(["Y"], axis=1)
            df = df.drop(["Z"], axis=1)
            if "D" in sort_by: df = df.drop(["D"], axis=1)
            if "DXY" in sort_by: df = df.drop(["DXY"], axis=1)
            if "DXZ" in sort_by: df = df.drop(["DXZ"], axis=1)
            if "DYZ" in sort_by: df = df.drop(["DYZ"], axis=1)
            if "DX" in sort_by: df = df.drop(["DX"], axis=1)
            if "DY" in sort_by: df = df.drop(["DY"], axis=1)
            if "DZ" in sort_by: df = df.drop(["DZ"], axis=1)
            return df

        cols = df.columns.tolist()
        if nf == 0:
            dff = pd.DataFrame(columns=cols)
            dfab = df.copy()
        else:
            dfab = df.truncate(after=df.index[na + nb - 1])
            dff = df.truncate(before=df.index[na + nb])
        if nb == 0:
            dfb = pd.DataFrame(columns=cols)
            dfa = dfab.copy()
        else:
            dfa = dfab.truncate(after=dfab.index[na - 1])
            dfb = dfab.truncate(before=dfab.index[na])

        dfa = df_sort(dfa, sort_by=self.sett.active_volume['Sort_by'],
                      sorting_spacer=self.sett.active_volume['SortingSpacer'],
                      sorting_shift=self.sett.active_volume['SortingShift'])
        if self.sett.active_volume['SortingBuffer']:
            dfb = df_sort(dfb, sort_by=self.sett.active_volume['Sort_by'],
                          sorting_spacer=self.sett.active_volume['SortingSpacer'],
                          sorting_shift=self.sett.active_volume['SortingShift'])
        if self.sett.active_volume['SortingFixed']:
            dff = df_sort(dff, sort_by=self.sett.active_volume['Sort_by'],
                          sorting_spacer=self.sett.active_volume['SortingSpacer'],
                          sorting_shift=self.sett.active_volume['SortingShift'])

        df = pd.concat([dfa, dfb, dff], axis=0)
        return df

    def get_avmask_inds(self, thiscoords, idc_reverse, grouped_atoms, rcutsq, rcut2sq, rcut3sq):
        thisidcxyz = np.multiply(thiscoords, self.cell_dim).astype(int) + 1
        for i in range(len(thisidcxyz)):
            if thisidcxyz[i] > self.cell_dim[i]: thisidcxyz[i] = self.cell_dim[i]
        thisncid = np.dot(np.add(cell_neigh_array, thisidcxyz), self.cell_dim_multiplier)
        xyzs = np.array([[], [], []])
        linds = np.array([], dtype=int)
        for cid in np.nditer(thisncid):
            ids = idc_reverse[cid]
            if ids >= 0:
                linds = np.hstack((linds, grouped_atoms[ids]["itag"]))
                xyzs = np.hstack((xyzs, np.vstack(
                    (grouped_atoms[ids]["xsn"], grouped_atoms[ids]["ysn"], grouped_atoms[ids]["zsn"]))))
        if linds.shape[0] > 0:
            thisxyznsq = np.sum(np.dot((xyzs.T - thiscoords), self.box.matrix) ** 2, axis=1)
            lindf = np.compress(thisxyznsq < rcut3sq, linds, axis=0)
            thisxyznsq = np.compress(thisxyznsq < rcut3sq, thisxyznsq, axis=0)
            linda = np.compress(thisxyznsq < rcutsq, lindf, axis=0)
            lindb = np.compress(thisxyznsq >= rcutsq, lindf, axis=0)
            thisxyznsq2 = np.compress(thisxyznsq >= rcutsq, thisxyznsq, axis=0)
            lindb = np.compress(thisxyznsq2 < rcut2sq, lindb, axis=0)
            lindf = np.compress(thisxyznsq >= rcut2sq, lindf, axis=0)
        else:
            linda = np.array([], dtype=int)
            lindb = np.array([], dtype=int)
            lindf = np.array([], dtype=int)
        return linda, lindb, lindf

    def get_this_av(self, atoms_array, masksa, masksb, masksf, keys, center, idav, Sorting=True):
        active = atoms_array[masksa]
        nactive = active.shape[0]
        idabfs = np.ones(nactive, dtype=int)

        buffer = atoms_array[masksb]
        nbuffer = buffer.shape[0]
        idabfs = np.concatenate((idabfs, np.zeros(nbuffer, dtype=int)), axis=0)

        fixed = atoms_array[masksf]
        nfixed = fixed.shape[0]
        idabfs = np.concatenate((idabfs, np.zeros(nfixed, dtype=int) - 1), axis=0)

        nt = nactive + nbuffer + nfixed
        if not isinstance(self.sett.active_volume["NMax4AV"], bool):
            if nt > self.sett.active_volume["NMax4AV"]:
                logstr = "Number of atoms in " + str(idav + 1) + " active volume is " + str(
                    nactive + nbuffer + nfixed) + "!"
                logstr += ("\n" +
                           "Maximum number of atoms in an active volume is " +
                           str(self.sett.active_volume["NMax4AV"]) + "!")
                logstr += "\n" + "Change active_volume['NMax4AV'] in input.yaml and rerun it!"
                raise ValueError(logstr)

        atoms_a = np.concatenate((active, buffer, fixed), axis=0)
        atoms = pd.DataFrame(atoms_a.T, columns=keys, index=np.arange(nt, dtype=int) + 1)
        atoms["idabf"] = idabfs
        atoms = self.df_center_atoms(atoms, Center=center, Style="Fractional")
        if Sorting:
            atoms = self.df_av_sort(atoms, nactive, nbuffer, nfixed)
        atoms = atoms.set_index(np.arange(len(atoms), dtype=int) + 1)
        itags = atoms["itag"].to_numpy().astype(int)
        idabfs = atoms["idabf"].to_numpy().astype(int)
        if nactive >= self.sett.active_volume["NMin4AV"]:
            atoms = atoms[ATOMS_HEADERS[self.atom_style]]
            av = ActiveVolume(idav, self.box, self.masses, atoms, itags, idabfs,
                              force_field=self.force_field,
                              topology=self.topology,
                              atom_style=self.atom_style,
                              nbuffer=nbuffer, nfixed=nfixed, sett=self.sett)
        else:
            logstr = "Number of active atoms in active volume is " + str(nactive) + "!"
            logstr += ("\n" +
                       "Minimum number of active atoms in an active volume is " +
                       str(self.sett.active_volume["NMin4AV"]) + "!")
            raise ValueError(logstr)
        return av

    def get_av(self, idf, Sorting=True):
        atoms_ghost_array = self.atoms_to_array(self.atoms_ghost, OutIndex=True)
        ntotcell = self.cell_dim_ghost[0] * self.cell_dim_ghost[1] * self.cell_dim_ghost[2]
        grouped_atoms, group_info, atomdtype = self.group_atoms_by("idc", atoms_ghost_array)
        keys = []
        for key in atomdtype.fields:
            keys.append(key)

        idc_reverse = np.array([-1] * ntotcell, dtype=int)
        for i in range(len(group_info[0])):
            idc_reverse[group_info[0][i]] = i

        atoms_array = self.atoms_to_array(self.atoms, OutIndex=True)

        rcut = self.sett.active_volume['DActive']
        rbuf = self.sett.active_volume['DBuffer']
        rfix = self.sett.active_volume['DFixed']
        rcutsq = rcut * rcut
        rcut2sq = (rcut + rbuf) * (rcut + rbuf)
        rcut3sq = (rcut + rbuf + rfix) * (rcut + rbuf + rfix)

        masksa = np.zeros(self.natoms, dtype=bool)
        masksb = np.zeros(self.natoms, dtype=bool)
        masksf = np.zeros(self.natoms, dtype=bool)
        nda = len(self.def_atoms[idf])
        if nda == 0: nda = 1
        for i in range(nda):
            if nda == 1:
                thiscoords = np.array(
                    [self.defects.iloc[idf]["xsn"], self.defects.iloc[idf]["ysn"], self.defects.iloc[idf]["zsn"]])
            else:
                thiscoords = np.array(
                    [self.def_atoms[idf][i]["xsn"], self.def_atoms[idf][i]["ysn"], self.def_atoms[idf][i]["zsn"]])
            thiscoords = np.subtract(thiscoords, thiscoords.round())
            thiscoords = np.select([thiscoords < 0, thiscoords < 1.0, thiscoords >= 1.0],
                                   [thiscoords + 1.0, thiscoords, thiscoords - 1.0])

            linda, lindb, lindf = self.get_avmask_inds(thiscoords, idc_reverse, grouped_atoms, rcutsq, rcut2sq, rcut3sq)
            if len(linda) > 0:
                masksa[linda] = True
                masksb[lindb] = True
                masksb[masksa] = False
                masksf[lindf] = True
                masksf[masksa] = False
                masksf[masksb] = False
        av = self.get_this_av(atoms_array, masksa, masksb, masksf,
                              keys, thiscoords, idf, Sorting=Sorting)
        return av

    def get_av_from_input(self, idf, nactive, nbuffer, nfixed, Sorting=True):
        atoms = self.atoms.copy()
        if nactive + nbuffer + nfixed < len(atoms):
            atoms = atoms.truncate(after=atoms.index[nactive + nbuffer + nfixed - 1])
        if "xsn" not in atoms.columns: atoms = self.get_fractional_coords(atoms)
        if "tag" not in atoms.columns: atoms = self.insert_tags(atoms)
        if "itag" not in atoms.columns: atoms = self.insert_itags(atoms)
        center = np.array([np.mean(atoms['xsn']), np.mean(atoms['ysn']), np.mean(atoms['zsn'])])
        atoms = self.insert_itags(atoms)
        atoms = atoms.set_index(np.arange(len(atoms), dtype=int) + 1)
        idabfs = np.ones(nactive, dtype=int)
        idabfs = np.concatenate((idabfs, np.zeros(nbuffer, dtype=int)), axis=0)
        idabfs = np.concatenate((idabfs, np.zeros(nfixed, dtype=int) - 1), axis=0)
        atoms["idabf"] = idabfs
        atoms = self.df_center_atoms(atoms, Center=center, Style="Fractional")
        if Sorting:
            atoms = self.df_av_sort(atoms, nactive, nbuffer, nfixed)
        atoms = atoms.set_index(np.arange(len(atoms), dtype=int) + 1)
        itags = atoms["itag"].to_numpy().astype(int)
        idabfs = atoms["idabf"].to_numpy().astype(int)
        if not isinstance(self.sett.active_volume["NMax4AV"], bool):
            if nactive > self.sett.active_volume["NMax4AV"]:
                logstr = "Number of atoms in " + str(idf + 1) + " active volume is " + str(nactive) + "!"
                logstr += "\n" + "Maximum number of atoms in an active volume is " + str(
                    self.sett.active_volume["NMax4AV"]) + "!"
                logstr += "\n" + "Change active_volume['NMax4AV'] in input.yaml and rerun it!"
                raise ValueError(logstr)
        if nactive >= self.sett.active_volume["NMin4AV"]:
            atoms = atoms[ATOMS_HEADERS[self.atom_style]]
            av = ActiveVolume(0, self.box, self.masses, atoms, itags, idabfs,
                              force_field=self.force_field,
                              topology=self.topology,
                              atom_style=self.atom_style,
                              nbuffer=nbuffer, nfixed=nfixed, sett=self.sett)
        else:
            logstr = "Number of active atoms in active volume is " + str(nactive) + "!"
            logstr += "\n" + "Minimum number of active atoms in an active volume is " + str(
                self.sett.active_volume["NMin4AV"]) + "!"
            raise ValueError(logstr)
        return av

    def get_active_volume(self, idf, Rebuild=True):
        #self.masses = self.sett.potential["masses"]
        if self.sett.active_volume['Style'].upper() == 'ALL':
            av = self.get_av_from_input(idf, self.natoms, 0, 0, Sorting=self.sett.active_volume['Sorting'])
        elif self.sett.active_volume['Style'].upper() == 'CUSTOM':
            av = self.get_av_from_input(idf, self.sett.active_volume['NActive'], self.sett.active_volume['NBuffer'],
                                        self.sett.active_volume['NFixed'],
                                        Sorting=self.sett.active_volume['Sorting'])
        else:
            if Rebuild:
                self.atoms = self.get_fractional_coords(self.atoms)
                self.atoms = self.insert_tags(self.atoms)
                self.atoms = self.insert_itags(self.atoms)
                rcut = self.sett.active_volume['DActive'] + self.sett.active_volume['DBuffer'] + \
                       self.sett.active_volume['DFixed']
                self.insert_atoms_cell(cellcut=rcut * 1.2)
            av = self.get_av(idf, Sorting=self.sett.active_volume['Sorting'])

        return av

    def update_coords_from_disps(self, idav, disps, AVitags, Reset_MolID=False):
        disps = disps.astype(float)
        for i in range(disps.shape[1]):
            itag = AVitags[idav][i]
            self.atoms.at[self.atoms.index[itag], "x"] += disps[0][i]
            self.atoms.at[self.atoms.index[itag], "y"] += disps[1][i]
            self.atoms.at[self.atoms.index[itag], "z"] += disps[2][i]
            if Reset_MolID:
                self.atoms.at[self.atoms.index[itag], "molecule-ID"] = int(Reset_MolID)

    def get_species_from_masses(self):
        masses = self.masses.copy()
        unique_masses = np.unique(masses["mass"])
        ref_masses = [el.atomic_mass.real for el in Element]
        diff = np.abs(np.array(ref_masses) - unique_masses[:, None])
        atomic_numbers = np.argmin(diff, axis=1) + 1
        symbols = [Element.from_Z(an).symbol for an in atomic_numbers]
        for um, s in zip(unique_masses, symbols):
            masses.loc[masses["mass"] == um, "element"] = s
        return masses["element"].to_numpy()

    def strings_from_file(self, filename):
        lines = []
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    lines.append(line.strip())
        return lines

    def to_strings(self, distance=6, velocity=8, charge=4):
        self.atoms = self.atoms[ATOMS_HEADERS[self.atom_style]]
        return self.get_string(distance=distance, velocity=velocity, charge=charge)

    def to_lammps_data(self, filename, to_atom_style=False, distance=6):
        if to_atom_style: self.to_atom_style()
        self.write_file(filename, distance=distance)

    def to_molecule(self, nout=None):
        if not isinstance(nout, int): nout = self.natoms
        try:
            species = np.array(self.sett.potential["species"])
        except:
            species = self.get_species_from_masses()
        types = self.atoms['type'].to_numpy().astype(int) - 1
        coords = np.vstack((self.atoms['x'], self.atoms['y'], self.atoms['z']))
        coords = coords.T
        types = types[0:nout]
        coords = coords[0:nout]
        eles = species[types]
        return Molecule(eles, coords)

    def to_structure(self):
        lattice = self.box.to_lattice()
        try:
            species = np.array(self.sett.potential["species"])
        except:
            species = self.get_species_from_masses()
        types = self.atoms['type'].to_numpy().astype(int) - 1
        coords = np.vstack((self.atoms['x'], self.atoms['y'], self.atoms['z']))
        coords = coords.T
        symbols = species[types]

        return Structure(lattice, symbols, coords,
                         coords_are_cartesian=True)

    def to_POSCAR(self, filename, Direct=False, SortElements=False):
        thisstructure = self.to_structure()
        if SortElements: thisstructure.sort()
        thisstructure.to(fmt='poscar', filename=filename, direct=Direct)

    def get_PG_OPs(self, nout=None):
        if not isinstance(nout, int): nout = self.natoms
        thisMol = self.to_molecule(nout=nout)
        PA = PointGroupAnalyzer(thisMol, tolerance=self.sett.system["Tolerance"])
        thisPGOPs = PA.get_symmetry_operations()
        return thisPGOPs, PA.sch_symbol

    def Write_AVs(self, filename, idavs="all", Buffer=False, Fixed=False, Invisible=True, Reset_Index=False):
        def get_thisidmol(masksa, masksb, masksf, Buffer=False, Fixed=False):
            if Buffer and Fixed:
                masks = masksa + masksb + masksf
            elif Buffer:
                masks = masksa + masksb
            else:
                masks = masksa.copy()
            return masks.astype(int)

        atoms = self.atoms.copy()
        atoms = atoms[ATOMS_HEADERS[self.atom_style]]
        newdata = SeakmcData(self.box, self.masses, atoms, force_field=self.force_field, topology=self.topology,
                             atom_style=self.atom_style, defects=self.defects, def_atoms=self.def_atoms, sett=self.sett)
        newdata.ndefects = len(newdata.defects)

        idmols = np.zeros(newdata.natoms, dtype=int)

        if isinstance(idavs, str) and idavs.upper() == "ALL":
            idavs = np.arange(newdata.ndefects, dtype=int)
        elif isinstance(idavs, list):
            idavs = np.array(idavs).astype(int)
        elif isinstance(idavs, np.ndarray):
            idavs = idavs.astype(int)
        else:
            try:
                idavs = np.array([int(idavs)], dtype=int)
            except:
                raise ValueError("idavs must be integer or list of integers or 'ALL'.")

        idavs = np.select([idavs < 0, idavs < newdata.ndefects, idavs >= newdata.ndefects],
                          [0, idavs, newdata.ndefects - 1])
        idavs = np.unique(idavs)
        rcut = newdata.sett.active_volume['DActive']
        rbuf = newdata.sett.active_volume['DBuffer']
        rfix = newdata.sett.active_volume['DFixed']

        newdata.atoms = newdata.get_fractional_coords(newdata.atoms)
        newdata.atoms = newdata.insert_itags(newdata.atoms)
        newdata.insert_atoms_cell(cellcut=(rcut + rbuf + rfix) * 1.2)

        atoms_ghost_array = newdata.atoms_to_array(newdata.atoms_ghost, OutIndex=True)
        ntotcell = newdata.cell_dim_ghost[0] * newdata.cell_dim_ghost[1] * newdata.cell_dim_ghost[2]
        grouped_atoms, group_info, atomdtype = newdata.group_atoms_by("idc", atoms_ghost_array)
        atoms_array = newdata.atoms_to_array(newdata.atoms, OutIndex=True)

        keys = []
        for key in atomdtype.fields:
            keys.append(key)

        idc_reverse = np.array([-1] * ntotcell, dtype=int)
        for i in range(len(group_info[0])):
            idc_reverse[group_info[0][i]] = i
            #itags = self.atoms["itag"].to_numpy().astype(int)

        for nd in range(idavs.shape[0]):
            idf = idavs[nd]
            rcutsq = rcut * rcut
            rcut2sq = (rcut + rbuf) * (rcut + rbuf)
            rcut3sq = (rcut + rbuf + rfix) * (rcut + rbuf + rfix)
            masksa = np.zeros(newdata.natoms, dtype=bool)
            masksb = np.zeros(newdata.natoms, dtype=bool)
            masksf = np.zeros(newdata.natoms, dtype=bool)
            nda = len(newdata.def_atoms[idf])
            if nda == 0: nda = 1
            for i in range(nda):
                if nda == 1:
                    thiscoords = np.array([newdata.defects.iloc[idf]["xsn"], newdata.defects.iloc[idf]["ysn"],
                                           newdata.defects.iloc[idf]["zsn"]])
                else:
                    thiscoords = np.array([newdata.def_atoms[idf][i]["xsn"], newdata.def_atoms[idf][i]["ysn"],
                                           newdata.def_atoms[idf][i]["zsn"]])
                thiscoords = np.subtract(thiscoords, thiscoords.round())
                thiscoords = np.select([thiscoords < 0, thiscoords >= 0], [thiscoords + 1.0, thiscoords])
                linda, lindb, lindf = newdata.get_avmask_inds(thiscoords, idc_reverse, grouped_atoms, rcutsq, rcut2sq,
                                                              rcut3sq)
                if linda.shape[0] > 0:
                    masksa[linda] = True
                    masksb[lindb] = True
                    masksb[masksa] = False
                    masksf[lindf] = True
                    masksf[masksa] = False
                    masksf[masksb] = False

            idmols += get_thisidmol(masksa, masksb, masksf, Buffer=Buffer, Fixed=Fixed)

        newdata.atoms = newdata.atoms.drop(["molecule-ID"], axis=1)
        im = ATOMS_HEADERS[self.atom_style].index('molecule-ID')
        newdata.atoms.insert(im, 'molecule-ID', idmols)

        if Invisible:
            atoms = newdata.atoms.drop(newdata.atoms[newdata.atoms["molecule-ID"] <= 0].index, inplace=False)
            newdata = SeakmcData(newdata.box, newdata.masses, atoms,
                                 force_field=newdata.force_field, topology=newdata.topology, atom_style=self.atom_style,
                                 sett=self.sett)

        if Reset_Index:
            newdata.atoms = newdata.atoms.set_index(np.arange(newdata.natoms, dtype=int) + 1)

        newdata.to_lammps_data(filename, to_atom_style=True, distance=self.sett.system["significant_figures"])

    def Write_Stack_SPs(self, SPlist, AVitags, fileheader, OutPath=False, rdcut4vis=0.01, dcut4vis=0.01, DispStyle="SP",
                        Invisible=True, Reset_Index=False):
        extra_cols = ["tag", "isp", "idv"]
        DispStyles = ["SPs", "FIs"]
        if DispStyle[0:2].upper() == "FI":
            istart = 1
            iend = 2
        elif DispStyle[0:2].upper() == "SP":
            istart = 0
            iend = 1
        else:
            istart = 0
            iend = 2

        for m in range(istart, iend):
            atoms = self.atoms.copy()
            atoms = atoms[ATOMS_HEADERS[self.atom_style]]
            natoms = len(atoms)
            thisdtype = self.generate_thisdtype(atoms, OutIndex=True)
            tags = atoms.index.to_numpy().astype(int)
            isps = np.zeros(natoms, dtype=int)
            idvs = np.zeros(natoms, dtype=int)
            for i in range(len(extra_cols)):
                if i == 0: atoms.insert(len(atoms.columns), extra_cols[i], tags)
                if i == 1: atoms.insert(len(atoms.columns), extra_cols[i], isps)
                if i == 2: atoms.insert(len(atoms.columns), extra_cols[i], idvs)

            id_add = 0
            for i in range(len(SPlist)):
                idav = SPlist[i].idav
                if m == 1:
                    thisdisp = SPlist[i].fdisp.copy()
                    thisdmax = SPlist[i].fdmax
                else:
                    thisdisp = SPlist[i].disp.copy()
                    thisdmax = SPlist[i].dmax
                thisdisp = thisdisp.astype(float)
                nout = thisdisp.shape[1]
                itags = AVitags[idav].copy()
                itags = itags[0:nout]
                thisds = np.sqrt(np.sum(thisdisp * thisdisp, axis=0))
                thisdcut = max(thisdmax * rdcut4vis, dcut4vis)

                inds = np.arange(nout, dtype=int)
                itags = np.compress(thisds > thisdcut, itags, axis=0)
                inds = np.compress(thisds > thisdcut, inds, axis=0)

                for ii in range(len(inds)):
                    id_add += 1
                    thisrow = self.atoms.loc[self.atoms.index[itags[ii]]].to_dict()
                    thisrow["x"] = thisrow["x"] + thisdisp[0][inds[ii]]
                    thisrow["y"] = thisrow["y"] + thisdisp[1][inds[ii]]
                    thisrow["z"] = thisrow["z"] + thisdisp[2][inds[ii]]
                    thisrow["tag"] = tags[itags[ii]]
                    thisrow["isp"] = i
                    thisrow["idv"] = i + 1
                    atoms.at[tags[itags[ii]], "isp"] += 1
                    atoms.at[tags[itags[ii]], "idv"] += i + 1
                    atoms.loc[self.idmax + id_add] = thisrow

            if Invisible: atoms = atoms.drop(atoms[atoms["idv"] <= 0].index, inplace=False)
            atoms[["molecule-ID", "type"]] = atoms[["molecule-ID", "type"]].astype(int)
            newdata = SeakmcData(self.box, self.masses, atoms,
                                 velocities=None, force_field=self.force_field, topology=self.topology,
                                 atom_style=self.atom_style, sett=self.sett)

            if Reset_Index: newdata.atoms = atoms.set_index(np.arange(len(atoms), dtype=int) + 1)
            filename = fileheader + DispStyles[m] + ".dat"
            if isinstance(OutPath, str): filename = os.path.join(OutPath, filename)
            newdata.to_lammps_data(filename, to_atom_style=False, distance=self.sett.system["significant_figures"])

    def Write_Separate_SPs(self, SPlist, AVitags, istep, OutPath=False, DispStyle="SP", Invisible=True, offset=0):
        DispStyles = ["SP", "FI"]
        if DispStyle[0:2].upper() == "FI":
            istart = 1
            iend = 2
        elif DispStyle[0:2].upper() == "SP":
            istart = 0
            iend = 1
        else:
            istart = 0
            iend = 2
        for m in range(istart, iend):
            for i in range(len(SPlist)):
                thisdata = copy.deepcopy(self)
                if m == 1:
                    thisdisp = SPlist[i].fdisp.copy()
                else:
                    thisdisp = SPlist[i].disp.copy()
                thisdisp = thisdisp.astype(float)
                thisdata.update_coords_from_disps(SPlist[i].idav, thisdisp, AVitags, Reset_MolID=i + 1)

                if Invisible:
                    thisdata.atoms = thisdata.atoms.drop(thisdata.atoms[thisdata.atoms["molecule-ID"] <= 0].index,
                                                         inplace=False)
                    thisdata.velocities = None
                    thisdata.natoms = len(thisdata.atoms)

            filename = "KMC_" + str(istep) + "_Data_" + DispStyles[m] + "_" + str(i + offset) + ".dat"
            if isinstance(OutPath, str): filename = os.path.join(OutPath, filename)
            thisdata.to_lammps_data(filename, to_atom_style=True, distance=self.sett.system["significant_figures"])
        thisdata = None

    def Write_Stack_SPs_from_DataSPs(self, iselSPs, DataSPs, AVitags, fileheader, OutPath=False, rdcut4vis=0.01,
                                     dcut4vis=0.01, DispStyle="SP", Invisible=True, Reset_Index=False):
        extra_cols = ["tag", "isp", "idv"]
        DispStyles = ["SPs", "FIs"]
        if DispStyle[0:2].upper() == "FI":
            istart = 1
            iend = 2
        elif DispStyle[0:2].upper() == "SP":
            istart = 0
            iend = 1
        else:
            istart = 0
            iend = 2

        for m in range(istart, iend):
            atoms = self.atoms.copy()
            atoms = atoms[ATOMS_HEADERS[self.atom_style]]
            natoms = len(atoms)
            thisdtype = self.generate_thisdtype(atoms, OutIndex=True)
            tags = atoms.index.to_numpy().astype(int)
            isps = np.zeros(natoms, dtype=int)
            idvs = np.zeros(natoms, dtype=int)
            for i in range(len(extra_cols)):
                if i == 0: atoms.insert(len(atoms.columns), extra_cols[i], tags)
                if i == 1: atoms.insert(len(atoms.columns), extra_cols[i], isps)
                if i == 2: atoms.insert(len(atoms.columns), extra_cols[i], idvs)

            id_add = 0
            for i in range(len(iselSPs)):
                isp = iselSPs[i]
                iavlocal = DataSPs.localiav[isp]
                isplocal = DataSPs.localisp[isp]
                idav = DataSPs.idavs[iavlocal]
                if m == 0:
                    thisdisp = DataSPs.disps[isp].copy()
                    thisdmax = DataSPs.df_SPs[iavlocal].at[isplocal, "dmax"]
                else:
                    thisdisp = DataSPs.fdisps[isp].copy()
                    thisdmax = DataSPs.df_SPs[iavlocal].at[isplocal, "fdmax"]
                thisdisp = thisdisp.astype(float)

                nout = thisdisp.shape[1]
                itags = AVitags[idav].copy()
                itags = itags[0:nout]
                thisds = np.sqrt(np.sum(thisdisp * thisdisp, axis=0))
                thisdcut = max(thisdmax * rdcut4vis, dcut4vis)

                inds = np.arange(nout, dtype=int)
                itags = np.compress(thisds > thisdcut, itags, axis=0)
                inds = np.compress(thisds > thisdcut, inds, axis=0)

                for ii in range(len(inds)):
                    id_add += 1
                    thisrow = self.atoms.loc[self.atoms.index[itags[ii]]].to_dict()
                    thisrow["x"] = thisrow["x"] + thisdisp[0][inds[ii]]
                    thisrow["y"] = thisrow["y"] + thisdisp[1][inds[ii]]
                    thisrow["z"] = thisrow["z"] + thisdisp[2][inds[ii]]
                    thisrow["tag"] = tags[itags[ii]]
                    thisrow["isp"] = isp
                    thisrow["idv"] = i + 1
                    atoms.at[tags[itags[ii]], "isp"] += isp
                    atoms.at[tags[itags[ii]], "idv"] += i + 1
                    atoms.loc[self.idmax + id_add] = thisrow

            if Invisible: atoms = atoms.drop(atoms[atoms["idv"] <= 0].index, inplace=False)
            atoms[["molecule-ID", "type"]] = atoms[["molecule-ID", "type"]].astype(int)
            newdata = SeakmcData(self.box, self.masses, atoms,
                                 velocities=None, force_field=self.force_field, topology=self.topology,
                                 atom_style=self.atom_style, sett=self.sett)

            if Reset_Index: newdata.atoms = atoms.set_index(np.arange(len(atoms), dtype=int) + 1)
            filename = fileheader + DispStyles[m] + ".dat"
            if isinstance(OutPath, str): filename = os.path.join(OutPath, filename)
            newdata.to_lammps_data(filename, to_atom_style=False, distance=self.sett.system["significant_figures"])

    def Write_Separate_SPs_from_DataSPs(self, iselSPs, DataSPs, AVitags, istep, OutPath=False, DispStyle="SP",
                                        Invisible=True, offset=0):
        DispStyles = ["SP", "FI"]
        if DispStyle[0:2].upper() == "FI":
            istart = 1
            iend = 2
        elif DispStyle[0:2].upper() == "SP":
            istart = 0
            iend = 1
        else:
            istart = 0
            iend = 2
        for m in range(istart, iend):
            for i in range(len(iselSPs)):
                isp = iselSPs[i]
                iavlocal = DataSPs.localiav[isp]
                idav = DataSPs.idavs[iavlocal]
                thisdata = copy.deepcopy(self)
                if m == 0:
                    thisdisp = DataSPs.disps[isp].copy()
                else:
                    thisdisp = DataSPs.fdisps[isp].copy()
                thisdisp = thisdisp.astype(float)
                thisdata.update_coords_from_disps(idav, thisdisp, AVitags, Reset_MolID=i + 1)

                if Invisible:
                    thisdata.atoms = thisdata.atoms.drop(thisdata.atoms[thisdata.atoms["molecule-ID"] <= 0].index,
                                                         inplace=False)
                    thisdata.velocities = None
                    thisdata.natoms = len(thisdata.atoms)

                filename = "KMC_" + str(istep) + "_Data_" + DispStyles[m] + "_" + str(isp + offset) + ".dat"
                if isinstance(OutPath, str): filename = os.path.join(OutPath, filename)
                thisdata.to_lammps_data(filename, to_atom_style=True, distance=self.sett.system["significant_figures"])
        thisdata = None

    def Write_SPs_from_Superbasin(self, isp, idav, thisBasin, istep, OutPath=False, DispStyle="SP", Invisible=True,
                                  offset=0):
        DispStyles = ["SP", "FI"]
        if DispStyle[0:2].upper() == "FI":
            istart = 1
            iend = 2
        elif DispStyle[0:2].upper() == "SP":
            istart = 0
            iend = 1
        else:
            istart = 0
            iend = 2
        for m in range(istart, iend):
            thisdata = copy.deepcopy(self)
            if m == 0:
                thisdisp = thisBasin.disps[isp].copy()
            else:
                thisdisp = thisBasin.fdisps[isp].copy()
            thisdisp = thisdisp.astype(float)
            thisdata.update_coords_from_disps(idav, thisdisp, thisBasin.AVitags, Reset_MolID=1)

            if Invisible:
                thisdata.atoms = thisdata.atoms.drop(thisdata.atoms[thisdata.atoms["molecule-ID"] <= 0].index,
                                                     inplace=False)
                thisdata.velocities = None
                thisdata.natoms = len(thisdata.atoms)

            filename = "KMC_" + str(istep) + "_Data_" + DispStyles[m] + "_" + str(isp + offset) + ".dat"
            if isinstance(OutPath, str): filename = os.path.join(OutPath, filename)
            thisdata.to_lammps_data(filename, to_atom_style=True, distance=self.sett.system["significant_figures"])
        thisdata = None

    def get_cart_from_fract(self, df):
        if "xsn" not in df.columns: df = self.get_fractional_coords(df)
        xyzns = np.vstack((df["xsn"], df["ysn"], df["zsn"]))
        xyzs = np.dot(xyzns.T, self.box.matrix)
        xyzs = xyzs.T
        df["x"] = xyzs[0]
        df["y"] = xyzs[1]
        df["z"] = xyzs[2]
        return df

    def modify_molecule_id(self, by, range=[0.0, 1.0], to_val=-2, Selection=False):
        by = by.upper()
        mol_id = self.atoms['molecule-ID'].to_numpy()
        if "D" in by:
            x = self.atoms['x'].to_numpy()
            y = self.atoms['y'].to_numpy()
            z = self.atoms['z'].to_numpy()
            if by == "DXYZ":
                v = np.sqrt(x * x + y * y + z * z)
            elif by == "DXY":
                v = np.sqrt(x * x + y * y)
            elif by == "DXZ":
                v = np.sqrt(x * x + z * z)
            elif by == "DYZ":
                v = np.sqrt(y * y + z * z)
        else:
            if by == "X":
                v = self.atoms['x'].to_numpy()
            elif by == "Y":
                v = self.atoms['y'].to_numpy()
            elif by == "Z":
                v = self.atoms['z'].to_numpy()
            elif by == "TYPE":
                v = self.atoms["type"].to_numpy()
            elif by == "XSN" or by == "YSN" or by == "ZSN":
                if "xsn" not in self.atoms.columns: self.atoms = self.get_fractional_coords(self.atoms)
                if by == "XSN":
                    v = self.atoms['xsn'].to_numpy()
                elif by == "YSN":
                    v = self.atoms['ysn'].to_numpy()
                elif by == "ZSN":
                    v = self.atoms['zsn'].to_numpy()

        if Selection:
            newid = np.select([v < range[0], v < range[1], v >= range[1]], [mol_id, to_val, mol_id])
        else:
            newid = np.select([v < range[0], v < range[1], v >= range[1]], [to_val, mol_id, to_val])
        self.atoms = self.atoms.drop(["molecule-ID"], axis=1)
        im = ATOMS_HEADERS[self.atom_style].index('molecule-ID')
        self.atoms.insert(im, 'molecule-ID', newid)

    def create_screw_dislocation(self, burgerv, burgerm, start_position, glide=0):
        burgerv = np.array(burgerv, dtype=int)
        ind = np.where(burgerv == 1)
        ind = ind[0][0]
        if ind == 0:
            x = self.atoms['y'] - start_position[1]
            y = self.atoms['z'] - start_position[2]
        elif ind == 1:
            x = self.atoms['x'] - start_position[0]
            y = self.atoms['z'] - start_position[2]
        else:
            x = self.atoms['x'] - start_position[0]
            y = self.atoms['y'] - start_position[1]
        if glide == 0:
            angles = np.arctan2(y, x)
        else:
            angles = np.arctan2(x, y)
        disps = -0.5 * burgerm * angles / pi
        if ind == 0:
            self.atoms['x'] += disps
        elif ind == 1:
            self.atoms['y'] += disps
        else:
            self.atoms['z'] += disps

    def modified_lmpbox(self, Operation="Translation", Values=[0, 0, 0], Ang_Format="DEGREE", Ang_Style="EU"):
        rotmat = None
        transvec = None

        if Operation[0:3].upper() == "ROT":
            rotmat = generate_rotation_matrix(Values, Ang_Format=Ang_Format, Ang_Style=Ang_Style)
        elif Operation[0:3].upper() == "TRA":
            transvec = np.array([Values[0], Values[1], Values[2]])
        if transvec is not None:
            self.box.bounds[0][1] += transvec[0]
            self.box.bounds[1][1] += transvec[1]
            self.box.bounds[2][1] += transvec[2]
        if rotmat is not None:
            if self.box.tilt is None: self.box.tilt = np.array([0, 0, 0])
            box1 = np.array([self.box.bounds[0][1] - self.box.bounds[0][0], 0, 0])
            box1 = np.dot(rotmat.T, box1)
            box2 = np.array([self.box.tilt[0], self.box.bounds[1][1] - self.box.bounds[1][0], 0])
            box2 = np.dot(rotmat.T, box2)
            box3 = np.array([self.box.tilt[1], self.box.tilt[2], self.box.bounds[1][1] - self.box.bounds[1][0]])
            box3 = np.dot(rotmat.T, box3)
            newbox = np.vstack([box1, box2, box3])
            newbox = to_half_matrix(newbox)
            self.box.bounds[0][0] = 0
            self.box.bounds[0][1] = newbox[0][0]
            self.box.bounds[1][0] = 0
            self.box.bounds[1][1] = newbox[1][1]
            self.box.bounds[2][0] = 0
            self.box.bounds[2][1] = newbox[2][2]
            self.box.tilt[0] = newbox[1][0]
            self.box.tilt[1] = newbox[2][0]
            self.box.tilt[2] = newbox[2][1]
        self.box = SeakmcBox(self.box.bounds, self.box.tilt)

    def chop_data(self, xlim=[0.25, 0.75], ylim=[0.25, 0.75], zlim=[0.25, 0.75], Fractional=True):
        if Fractional:
            self.atoms = self.get_fractional_coords(self.atoms)
            self.atoms = self.atoms[
                (self.atoms["xsn"] >= xlim[0]) & (self.atoms["xsn"] < xlim[1]) & (self.atoms["ysn"] >= ylim[0]) & (
                            self.atoms["ysn"] < ylim[1]) & (self.atoms["zsn"] >= zlim[0]) & (
                            self.atoms["zsn"] < zlim[1])]
            #self.atoms = self.atoms[(self.atoms["ysn"]>=ylim[0]) & (self.atoms["ysn"]<ylim[1])]
            #self.atoms = self.atoms[(self.atoms["zsn"]>=zlim[0]) & (self.atoms["zsn"]<zlim[1])]
        else:
            self.atoms = self.atoms[
                (self.atoms["x"] >= xlim[0]) & (self.atoms["x"] < xlim[1]) & (self.atoms["y"] >= ylim[0]) & (
                            self.atoms["y"] < ylim[1]) & (self.atoms["z"] >= zlim[0]) & (self.atoms["z"] < zlim[1])]
            #self.atoms = self.atoms[(self.atoms["y"]>=ylim[0]) & (self.atoms["y"]<ylim[1])]
            #self.atoms = self.atoms[(self.atoms["z"]>=zlim[0]) & (self.atoms["z"]<zlim[1])]
        self.natoms = len(self.atoms)

    def chop_data_by(self, by="type", vlim=[0, 1], Selection=True):
        if Selection:
            self.atoms = self.atoms[(self.atoms[by] >= vlim[0]) & (self.atoms[by] < vlim[1])]
        else:
            self.atoms = self.atoms[(self.atoms[by] <= vlim[0]) | (self.atoms[by] > vlim[1])]
        self.natoms = len(self.atoms)

    def shear_atoms(self, strainrate, shear_ref=1, shear_dir=2,
                    start_position="NA", end_position="NA", Centered=True, Abs_dir=False, isFixed=False,
                    ref_position=0.0, Abs_ref=False):
        if "xsn" not in self.atoms.columns: self.atoms = self.get_fractional_coords(self.atoms)
        if shear_ref == 0:
            x = self.atoms['x'].to_numpy()
        elif shear_ref == 1:
            x = self.atoms['y'].to_numpy()
        else:
            x = self.atoms['z'].to_numpy()

        thismask = np.ones(self.natoms, dtype=bool)
        if isinstance(start_position, str): start_position = np.min(x)
        if isinstance(end_position, str): end_position = np.max(x)
        thisstart = min(start_position, end_position)
        thisend = max(start_position, end_position)
        thismask = np.select([x < thisstart, x <= thisend, x > thisend], [0, thismask, 0])

        if isFixed:
            dy = np.array(self.natoms * [strainrate])
            dy = np.select([thismask == 0, thismask == 1], [0.0, dy])
        else:
            x = x - ref_position
            if Abs_ref: x = np.absolute(x)
            dy = x * strainrate
            dy = np.select([thismask == 0, thismask == 1], [0.0, dy])

        if Centered:
            if shear_dir == 0:
                thisfract = self.atoms['xsn'].to_numpy() - 0.5
            elif shear_dir == 1:
                thisfract = self.atoms['ysn'].to_numpy() - 0.5
            else:
                thisfract = self.atoms['zsn'].to_numpy() - 0.5

            if Abs_dir:
                thisfract = np.absolute(thisfract)
                thisscale = 0.5 / np.max(thisfract)
                thisfract = thisfract * thisscale
            else:
                thisscale = 0.5 / np.max(np.absolute(thisfract))
                thisfract = thisfract * thisscale
            dy = thisfract * dy

        if shear_dir == 0:
            self.atoms['x'] += dy
        elif shear_dir == 1:
            self.atoms['y'] += dy
        else:
            self.atoms['z'] += dy

        self.atoms = self.get_fractional_coords(self.atoms)

    def shear_box(self, strainrate, shear_ref=1, shear_dir=2, move_atoms=True):
        if "xsn" not in self.atoms.columns: self.atoms = self.get_fractional_coords(self.atoms)
        if shear_ref == 0:
            x = self.box.bounds[0][1] - self.box.bounds[0][0]
        elif shear_ref == 1:
            x = self.box.bounds[1][1] - self.box.bounds[1][0]
        else:
            x = self.box.bounds[2][1] - self.box.bounds[2][0]
        dy = x * strainrate
        if self.box.tilt is None: self.box.tilt = np.array([0.0, 0.0, 0.0])
        if shear_ref == 0 and shear_dir == 1:
            self.box.tilt[0] += dy
        elif shear_ref == 0 and shear_dir == 2:
            self.box.tilt[1] += dy
        elif shear_ref == 1 and shear_dir == 0:
            self.box.tilt[0] -= dy
        elif shear_ref == 1 and shear_dir == 2:
            self.box.tilt[2] += dy
        elif shear_ref == 2 and shear_dir == 0:
            self.box.tilt[1] -= dy
        elif shear_ref == 2 and shear_dir == 1:
            self.box.tilt[2] -= dy
        else:
            pass
        self.box = SeakmcBox(self.box.bounds, self.box.tilt)
        if move_atoms:
            self.atoms = self.get_cart_from_fract(self.atoms)

    def extract_neighbor_list_lmp(self, lmp, idx):
        neighbor_list = lmp.numpy.get_neighlist(idx)
        nlocal = lmp.extract_global("nlocal")
        nghost = lmp.extract_global("nghost")
        tags = lmp.extract_atom("id")
        tags = np.ctypeslib.as_array(tags, shape=(nlocal + nghost,))
        neighbors = {"tags": tags, "neighlist": neighbor_list}
        return neighbors


class ActiveVolume(SeakmcData, MSONable):
    def __init__(
            self,
            idav,
            box,
            masses,
            atoms,
            itags,
            idabfs,
            force_field=None,
            topology=None,
            atom_style="full",
            nbuffer=0,
            nfixed=0,
            cusatoms=None,
            sett=None,
    ):
        self.idav = idav
        self.itags = itags
        self.idabfs = idabfs
        box = SeakmcBox(box.bounds, box.tilt)
        #atoms = atoms[ATOMS_HEADERS[atom_style]]
        super().__init__(
            box,
            masses,
            atoms,
            velocities=None,
            force_field=force_field,
            topology=topology,
            atom_style=atom_style,
            cusatoms=cusatoms,
            sett=sett,
        )

        self.natoms = len(self.atoms)
        self.nbuffer = nbuffer
        self.nfixed = nfixed
        self.nactive = self.natoms - self.nbuffer - self.nfixed
        if self.nactive == 0:
            warnings.warn("Warning: there is no active atom in " + str(self.idav) + " AV!")
        self.nactbuf = self.natoms - self.nfixed
        self.cusatoms = cusatoms
        if sett is None:
            self.sett = Global_Variables
        else:
            self.sett = sett

        self.idmax = self.atoms.index.max()
        self.dimension = 3
        self.PBC = [True, True, True]
        try:
            self.dimension = self.sett.data["dimension"]
        except:
            pass
        try:
            self.PBC = copy.deepcopy(self.sett.data["PBC"])
        except:
            pass

        try:
            if self.sett.active_volume["ResetBounds"]:
                if self.sett.active_volume["TurnoffPBC"][0]:
                    self.box.bounds[0][0] = np.min(self.atoms["x"]) - 1.0
                    self.box.bounds[0][1] = np.max(self.atoms["x"]) + 1.0
                    self.box.tilt[0] = 0.0
                    self.box.tilt[1] = 0.0
                    self.PBC[0] = False
                if self.sett.active_volume["TurnoffPBC"][1]:
                    self.box.bounds[1][0] = np.min(self.atoms["y"]) - 1.0
                    self.box.bounds[1][1] = np.max(self.atoms["y"]) + 1.0
                    self.box.tilt[0] = 0.0
                    self.box.tilt[2] = 0.0
                    self.PBC[1] = False
                if self.sett.active_volume["TurnoffPBC"][2]:
                    self.box.bounds[2][0] = np.min(self.atoms["z"]) - 1.0
                    self.box.bounds[2][1] = np.max(self.atoms["z"]) + 1.0
                    self.box.tilt[1] = 0.0
                    self.box.tilt[2] = 0.0
                    self.PBC[2] = False
                self.box = SeakmcBox(self.box.bounds, self.box.tilt)
        except:
            pass

    def __str__(self):
        return "Active volume id is ({}).".format(self.idav)

    def __repr__(self):
        return self.__str__()

    def to_coords(self, Buffer=False, Fixed=False):
        nout = self.nactive
        if Buffer: nout += self.nbuffer
        if Fixed: nout += self.nfixed
        x = self.atoms["x"].to_numpy()
        x = x[0:nout]
        y = self.atoms["y"].to_numpy()
        y = y[0:nout]
        z = self.atoms["z"].to_numpy()
        z = z[0:nout]
        this_coords = np.vstack((x, y, z))
        return this_coords

    def assign_idabf2mol(self):
        if "molecule-ID" in self.atoms.columns:
            self.atoms = self.atoms.drop(["molecule-ID"], axis=1)
        else:
            if self.atom_style == "atomic": self.atom_style = "molecular"
            if self.atom_style == "charge": self.atom_style = "full"
        im = ATOMS_HEADERS[self.atom_style].index('molecule-ID')
        self.atoms.insert(im, 'molecule-ID', self.idabfs)

    def to_saddle_point(self, coords):
        if isinstance(coords, np.ndarray):
            n = coords.shape[1]
            v = self.atoms['x'].to_numpy()
            v = np.hstack((coords[0], v[n:self.natoms]))
            self.atoms = self.atoms.drop(['x'], axis=1)
            im = ATOMS_HEADERS[self.atom_style].index('x')
            self.atoms.insert(im, 'x', v)

            v = self.atoms['y'].to_numpy()
            v = np.hstack((coords[1], v[n:self.natoms]))
            self.atoms = self.atoms.drop(['y'], axis=1)
            im = ATOMS_HEADERS[self.atom_style].index('y')
            self.atoms.insert(im, 'y', v)

            v = self.atoms['z'].to_numpy()
            v = np.hstack((coords[2], v[n:self.natoms]))
            self.atoms = self.atoms.drop(['z'], axis=1)
            im = ATOMS_HEADERS[self.atom_style].index('z')
            self.atoms.insert(im, 'z', v)

            if isinstance(self.cusatoms, pd.DataFrame):
                self.insert_cusatoms(Sort_by='type', Ascending=True)
        else:
            pass

    def update_avcoords_from_disps(self, disps):
        if isinstance(disps, np.ndarray):
            disps = disps.astype(float)
            n = disps.shape[1]
            zerosapp = np.zeros((3, self.natoms - n), dtype=float)
            thisdisps = np.hstack((disps, zerosapp))
            self.atoms['x'] = np.add(self.atoms['x'].to_numpy(), thisdisps[0])
            self.atoms['y'] = np.add(self.atoms['y'].to_numpy(), thisdisps[1])
            self.atoms['z'] = np.add(self.atoms['z'].to_numpy(), thisdisps[2])
        else:
            pass

    def update_coords(self, coords):
        if isinstance(coords, np.ndarray):
            coords = coords.astype(float)
            n = coords.shape[1]
            self.atoms.loc[:n, 'x'] = coords[0]
            self.atoms.loc[:n, 'y'] = coords[1]
            self.atoms.loc[:n, 'z'] = coords[2]
        else:
            pass

    def estimate_atom_strain(self, thissett, nactive=None, nbuffer=None):
        if nactive is None:
            nactive = self.nactive
        else:
            nactive = min(nactive, self.natoms)
        if nbuffer is None:
            nbuffer = self.nbuffer
        else:
            nbuffer = min(nbuffer, self.natoms - nactive)
        cutdefectmax = thissett.active_volume['cutdefectmax']
        cutdefectmaxsq = cutdefectmax * cutdefectmax
        df = self.atoms.truncate(after=self.atoms.index[nactive + nbuffer - 1])
        df = self.get_fractional_coords(df)
        ##df = self.insert_itags(df)
        df = self.insert_df_cell(df, nactive + nbuffer, cellcut=cutdefectmax * 1.2)
        atoms_ghost_array = self.atoms_to_array(df, OutIndex=True)

        ntotcell = self.cell_dim_ghost[0] * self.cell_dim_ghost[1] * self.cell_dim_ghost[2]
        grouped_atoms, group_info, atomdtype = self.group_atoms_by("idc", atoms_ghost_array)

        idc_reverse = np.array([-1] * ntotcell, dtype=int)
        for i in range(len(group_info[0])):
            idc_reverse[group_info[0][i]] = i

        strain_array = np.zeros(nactive, dtype=float)
        zerocn = []
        for nr in range(nactive):
            thiscoords = np.array([df.iloc[nr]["xsn"], df.iloc[nr]["ysn"], df.iloc[nr]["zsn"]])
            thistype = int(df.iloc[nr]["type"]) - 1
            thisatoms = self.get_cell_atoms(nr, thiscoords, idc_reverse, grouped_atoms, atomdtype, Self_Excluded=True,
                                            isHalf=False, indkey="index")

            inds = np.array(thisatoms["index"], dtype=int)
            xyzs = np.vstack((thisatoms["xsn"], thisatoms["ysn"], thisatoms["zsn"]))
            types = np.array(thisatoms["type"], dtype=int)

            if inds.shape[0] <= 0:
                zerocn.append(nr)
            else:
                #thisxyznsq = np.sum((xyzs.T - thiscoords)**2, axis = 1)
                thisxyznsq = np.sum(np.dot((xyzs.T - thiscoords), self.box.matrix) ** 2, axis=1)
                inds = np.compress(thisxyznsq < cutdefectmaxsq, inds, axis=0)
                types = np.compress(thisxyznsq < cutdefectmaxsq, types, axis=0)
                thisxyznsq = np.compress(thisxyznsq < cutdefectmaxsq, thisxyznsq, axis=0)
                types = types - 1
                masks = np.zeros(inds.shape, dtype=bool)
                for j in range(inds.shape[0]):
                    thiscut = thissett.potential['cutneighs4LAS'][thistype][types[j]]
                    thiscutsq = thiscut * thiscut
                    if thisxyznsq[j] < thiscutsq: masks[j] = True

                thisxyznsq = thisxyznsq[masks]
                types = types[masks]
                inds = inds[masks]
                thisxyznsq = np.sqrt(thisxyznsq)
                cntype = thissett.potential['coordnums4LAS'][thistype]
                thisscale = (1.0 - 0.2 * (cntype - inds.shape[0]) / cntype)
                thisscale = min(thisscale, 1.5)
                thisscale = max(thisscale, 0.5)
                for j in range(inds.shape[0]):
                    bl = thissett.potential['bondlengths4LAS'][thistype][types[j]] * thisscale
                    strain_array[nr] += 100 * (thisxyznsq[j] - bl) / bl / cntype

        meanv = np.mean(strain_array)
        strain_array = np.absolute(strain_array - meanv)
        #strain_array = strain_array-meanv
        if len(zerocn) > 0:
            strainmax = np.max(strain_array)
            for i in zerocn:
                strain_array[i] = strainmax
        self.atoms_ghost = None
        self.natoms_ghost = 0
        dropcols = []
        if "xsn" in self.atoms.columns: dropcols.append("xsn")
        if "ysn" in self.atoms.columns: dropcols.append("ysn")
        if "zsn" in self.atoms.columns: dropcols.append("zsn")
        if "idc" in self.atoms.columns: dropcols.append("idc")
        if "itag" in self.atoms.columns: dropcols.append("itag")
        if len(dropcols) > 0: self.atoms = self.atoms.drop(dropcols, axis=1)
        return strain_array

    def generate_displacements(self, dispsett):
        rotmat = None
        transvec = None
        disps = None
        if DispSettKEY[2] in dispsett and DispSettKEY[3] in dispsett:
            xyzs = np.vstack(
                (self.atoms['x'][0:self.nactive], self.atoms['y'][0:self.nactive], self.atoms['z'][0:self.nactive]))
            if dispsett[DispSettKEY[2]][0:3].upper() == "ROT":
                alpha = dispsett[DispSettKEY[3]][0] * pi / 180.0
                beta = dispsett[DispSettKEY[3]][1] * pi / 180.0
                gamma = dispsett[DispSettKEY[3]][2] * pi / 180.0
                rotmat = generate_rotation_matrix([alpha, beta, gamma], Ang_Format="Radian", Ang_Style="Euler")
            elif dispsett[DispSettKEY[2]][0:3].upper() == "TRA":
                transvec = np.array(
                    [dispsett[DispSettKEY[3]][0], dispsett[DispSettKEY[3]][1], dispsett[DispSettKEY[3]][2]])

            if isinstance(dispsett[DispSettKEY[0]], str):
                if dispsett[DispSettKEY[0]].upper() == "ALL":
                    newxyzs = copy.deepcopy(xyzs)
                    newxyzs = newxyzs.T
                    if transvec is not None:
                        newxyzs += transvec
                    if rotmat is not None:
                        newxyzs = np.dot(newxyzs, rotmat)
                    disps = newxyzs.T - xyzs

            elif isinstance(dispsett[DispSettKEY[0]], list):
                nentry = len(dispsett[DispSettKEY[0]])
                if nentry != len(dispsett[DispSettKEY[1]]):
                    pass
                else:
                    nval_entry = 0
                    ind_entry = []
                    minvals = []
                    maxvals = []
                    for ien in range(nentry):
                        try:
                            thisind = EntryKEY.index(dispsett[DispSettKEY[0]][ien])
                            nval_entry += 1
                            ind_entry.append(thisind)
                            minvals.append(dispsett[DispSettKEY[1]][ien][0])
                            maxvals.append(dispsett[DispSettKEY[1]][ien][1])
                        except:
                            pass

                    newxyzs = copy.deepcopy(xyzs)
                    for nr in range(xyzs.shape[1]):
                        thisxyz = np.array([xyzs[0][nr], xyzs[1][nr], xyzs[2][nr]])
                        isSel = True
                        for ien in range(len(ind_entry)):
                            indi = ind_entry[ien]
                            if indi == 0:
                                thisval = self.atoms.iloc[nr]["type"]
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                            elif indi == 1:
                                thisval = self.atoms.index[nr]
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                            elif indi == 2:
                                thisval = thisxyz[0]
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                            elif indi == 3:
                                thisval = thisxyz[1]
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                            elif indi == 4:
                                thisval = thisxyz[2]
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                            elif indi == 5:
                                thisval = np.sqrt(thisxyz[0] ** 2 + thisxyz[1] ** 2)
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                            elif indi == 6:
                                thisval = np.sqrt(thisxyz[0] ** 2 + thisxyz[2] ** 2)
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                            elif indi == 7:
                                thisval = np.sqrt(thisxyz[1] ** 2 + thisxyz[2] ** 2)
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                            elif indi == 8:
                                thisval = np.sqrt(thisxyz[0] ** 2 + thisxyz[1] ** 2 + thisxyz[2] ** 2)
                                if thisval < minvals[ien] or thisval > maxvals[ien]:
                                    isSel = False
                                    break
                        if isSel:
                            if transvec is not None:
                                thisxyz += transvec
                            if rotmat is not None:
                                thisxyz = np.dot(rotmat.T, thisxyz)
                            newxyzs[0][nr] = thisxyz[0]
                            newxyzs[1][nr] = thisxyz[1]
                            newxyzs[2][nr] = thisxyz[2]
                    disps = newxyzs - xyzs
        return disps

    def Write_Stack_avSPs(self, SPlist, fileheader, OutPath=False, rdcut4vis=0.01, dcut4vis=0.01, DispStyle="SP",
                        Invisible=True, Reset_Index=False):
        extra_cols = ["tag", "isp", "idv"]
        DispStyles = ["SPs", "FIs"]
        if DispStyle[0:2].upper() == "FI":
            istart = 1
            iend = 2
        elif DispStyle[0:2].upper() == "SP":
            istart = 0
            iend = 1
        else:
            istart = 0
            iend = 2
        for m in range(istart, iend):
            atoms = self.atoms.copy()
            atoms = atoms[ATOMS_HEADERS[self.atom_style]]
            thisdtype = self.generate_thisdtype(atoms, OutIndex=True)
            tags = atoms.index.to_numpy().astype(int)
            isps = np.zeros(len(atoms), dtype=int)
            idvs = np.zeros(len(atoms), dtype=int)
            for i in range(len(extra_cols)):
                if i == 0: atoms.insert(len(atoms.columns), extra_cols[i], tags)
                if i == 1: atoms.insert(len(atoms.columns), extra_cols[i], isps)
                if i == 2: atoms.insert(len(atoms.columns), extra_cols[i], idvs)

            id_add = 0
            for i in range(len(SPlist)):
                if m == 1:
                    thisdisp = SPlist[i].fdisp.copy()
                    thisdmax = SPlist[i].fdmax
                else:
                    thisdisp = SPlist[i].disp.copy()
                    thisdmax = SPlist[i].dmax
                thisdisp = thisdisp.astype(float)

                nout = thisdisp.shape[1]
                thisds = np.sqrt(np.sum(thisdisp * thisdisp, axis=0))
                thisdcut = max(thisdmax * rdcut4vis, dcut4vis)
                inds = np.arange(nout, dtype=int)
                inds = np.compress(thisds > thisdcut, inds, axis=0)
                for ii in range(len(inds)):
                    id_add += 1
                    thisrow = self.atoms.loc[self.atoms.index[inds[ii]]].to_dict()
                    thisrow["x"] = thisrow["x"] + thisdisp[0][inds[ii]]
                    thisrow["y"] = thisrow["y"] + thisdisp[1][inds[ii]]
                    thisrow["z"] = thisrow["z"] + thisdisp[2][inds[ii]]
                    thisrow["tag"] = tags[inds[ii]]
                    thisrow["isp"] = i
                    thisrow["idv"] = i + 1
                    atoms.at[tags[inds[ii]], "isp"] += 1
                    atoms.at[tags[inds[ii]], "idv"] += i + 1
                    atoms.loc[self.idmax + id_add] = thisrow

            if Invisible: atoms = atoms.drop(atoms[atoms["idv"] <= 0].index, inplace=False)
            atoms[["molecule-ID", "type"]] = atoms[["molecule-ID", "type"]].astype(int)

            newdata = ActiveVolume(self.idav, self.box, self.masses, atoms, np.arange(len(atoms), dtype=int),
                                   np.zeros(len(atoms), dtype=int),
                                   force_field=self.force_field, topology=self.topology, atom_style=self.atom_style)
            if Reset_Index: newdata.atoms = newdata.atoms.set_index(np.arange(len(atoms), dtype=int) + 1)
            filename = fileheader + DispStyles[m] + ".dat"
            if isinstance(OutPath, str): filename = os.path.join(OutPath, filename)
            newdata.to_lammps_data(filename, to_atom_style=False, distance=self.sett.system["significant_figures"])


class ActiveVolumeSPS(ActiveVolume):
    def __init__(
            self,
            idsps,
            idav,
            box,
            masses,
            atoms,
            itags,
            idabfs,
            force_field=None,
            topology=None,
            atom_style="full",
            nbuffer=0,
            nfixed=0,
            cusatoms=None,
            sett=None,
    ):
        self.idsps = idsps

        super().__init__(
            idav,
            box,
            masses,
            atoms,
            itags,
            idabfs,
            force_field=force_field,
            topology=topology,
            atom_style=atom_style,
            nbuffer=nbuffer,
            nfixed=nfixed,
            cusatoms=cusatoms,
            sett=sett,
        )

    def __str__(self):
        return "This SPS id is ({}) and active volume id is ({}).".format(self.idsps, self.idav)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_activevolume(cls, idsps, av):
        thisatoms = av.atoms.copy()
        if isinstance(av.cusatoms, pd.DataFrame):
            thiscusatoms = av.cusatoms.copy()
        else:
            thiscusatoms = None
        return (cls(idsps, av.idav, av.box, av.masses, thisatoms,
                    av.itags, av.idabfs,
                    force_field=av.force_field,
                    topology=av.topology,
                    atom_style=av.atom_style,
                    nbuffer=av.nbuffer,
                    nfixed=av.nfixed,
                    cusatoms=thiscusatoms,
                    sett=av.sett, ))

    def get_disp_lattice(self, coords, coords_org=None):
        """
        get displacement lattice (3*3)
        """
        nactive = int(coords.size / 3)
        if not isinstance(coords_org, np.ndarray):
            coords_org = np.vstack((self.atoms['x'][0:nactive], self.atoms['y'][0:nactive], self.atoms['z'][0:nactive]))
        return np.divide(np.dot(coords_org, coords.T), nactive)

    def get_disp_mat(self, coords, coords_org=None):
        nactive = int(coords.size / 3)
        if not isinstance(coords_org, np.ndarray):
            coords_org = np.vstack((self.atoms['x'][0:nactive], self.atoms['y'][0:nactive], self.atoms['z'][0:nactive]))
        return np.dot(coords - coords_org, coords_org.T)
