"""Residual-depth-moveout velocity updates on 2D common-image gathers.

The depth curve is z(h)^2=z0^2+a*h^2, where h is half-offset and, for a horizontal
reflector in a constant medium, a=(v_migration/v_true)^2-1. The vertical update
uses this exact constant-medium relation. The tomographic option linearises it
with straight two-leg rays and a zero-offset depth-adjustment term. It is a
small-contrast approximation; it is not bent-ray reflection tomography.
"""
import numpy as np
from scipy.sparse import csr_matrix, diags, vstack

from ..core.tomography import damped_update, roughness_operators
from ..core.velocity import DepthModel
from .base import Param, Process, register
from .fwi import model_section


def scan_depth_moveout(traces, depths, offsets, targets, coefficients, window=20.0):
    g, z, offsets = np.asarray(traces), np.asarray(depths), np.asarray(offsets)
    live = np.max(np.abs(g), axis=1) > 0
    g, offsets = g[live], offsets[live]
    if len(g) < 3 or np.ptp(np.abs(offsets)) == 0:
        return []
    shifts = np.arange(-window / 2, window / 2 + 0.1 * (z[1] - z[0]), z[1] - z[0])
    results = []
    peak = float(np.max(np.abs(g)))
    for depth in targets:
        if depth <= 0:
            continue
        scores, energies = [], []
        for a in coefficients:
            z2 = (depth + shifts[None, :]) ** 2 + a * (offsets[:, None] / 2) ** 2
            zz = np.sqrt(np.maximum(z2, 0))
            sample = np.array([np.interp(zz[i], z, row, left=0, right=0) for i, row in enumerate(g)])
            sample[z2 <= 0] = 0
            energy = float(np.sum(sample ** 2))
            scores.append(float(np.sum(sample.sum(axis=0) ** 2) / max(len(g) * energy, 1e-30)))
            energies.append(float(np.max(np.abs(sample))))
        k = int(np.argmax(scores))
        if energies[k] > 0.1 * peak and scores[k] > 0.3 and 0 < k < len(coefficients) - 1:
            results.append({"depth": float(depth), "a": float(coefficients[k]), "semblance": scores[k]})
    return results


def straight_ray(model, x_start, x_end, z_end):
    """Bilinear nodal path lengths, with endpoint-aware integration on a straight ray."""
    length = float(np.hypot(x_end - x_start, z_end - model.z0))
    if min(x_start, x_end) < model.x0 or max(x_start, x_end) > model.x[-1] or z_end > model.z[-1]:
        raise ValueError("image-gather aperture extends outside the project velocity model")
    n = max(1, int(np.ceil(length / (0.5 * min(model.dx, model.dz)))))
    f = (np.arange(n) + 0.5) / n
    x = (x_start + f * (x_end - x_start) - model.x0) / model.dx
    z = f * (z_end - model.z0) / model.dz
    ix = np.clip(x.astype(int), 0, model.shape[1] - 2)
    iz = np.clip(z.astype(int), 0, model.shape[0] - 2)
    fx, fz = x - ix, z - iz
    columns, weights = [], []
    for ox, oz, w in ((0, 0, (1-fx)*(1-fz)), (1, 0, fx*(1-fz)), (0, 1, (1-fx)*fz), (1, 1, fx*fz)):
        columns.extend(((iz + oz) * model.shape[1] + ix + ox).tolist())
        weights.extend((w * length / n).tolist())
    return csr_matrix((weights, (np.zeros(len(columns), dtype=int), columns)), shape=(1, model.values.size))


@register
class MigrationVelocityAnalysis(Process):
    name, label, category = "mva_tomography", "Residual-moveout velocity update (vertical / straight-ray tomography)", "Tomography"
    description = ("Scan residual depth moveout on Kirchhoff common-image gathers. Update a project velocity model "
                   "vertically or by damped straight-ray differential tomography, then re-migrate the original data "
                   "to assess the change. Small velocity errors, locally flat reflectors and 2D geometry only.")
    params = (
        Param("method", "choice", "vertical", "Update", choices=("vertical", "straight-ray tomography")),
        Param("zmin", "float", 100.0, "First analysis depth", unit="m", minimum=1.0),
        Param("zmax", "float", 2000.0, "Last analysis depth", unit="m", minimum=1.0),
        Param("depth_step", "float", 20.0, "Analysis depth interval", unit="m", minimum=1.0),
        Param("window", "float", 30.0, "Semblance depth window", unit="m", minimum=1.0),
        Param("amax", "float", 0.3, "Maximum absolute moveout coefficient", minimum=0.01, maximum=0.8),
        Param("na", "int", 61, "Moveout coefficients", minimum=5, maximum=401),
        Param("smoothing", "float", 2.0, "Tomographic smoothing", minimum=0.0),
        Param("damping", "float", 0.2, "Tomographic damping", minimum=0.01),
        Param("max_change", "float", 0.1, "Maximum fractional velocity update", minimum=0.001, maximum=0.3),
    )

    def run(self, data, context):
        if (data.domain != "depth" or not all(k in data.headers for k in ("offset", "cdp", "cdpx"))
                or context.depth_model is None):
            raise ValueError("MVA needs depth image gathers with offsets and the model used to migrate them")
        model = context.depth_model
        if model.z0 != 0:
            raise ValueError("this RMO approximation requires a model starting at the flat datum z=0")
        coefficients = np.linspace(-self["amax"], self["amax"], self["na"])
        targets = np.arange(self["zmin"], min(self["zmax"], data.tmax) + 1e-9, self["depth_step"])
        picks, matrices, residuals = [], [], []
        updates, positions = [], []
        groups = list(data.iter_gathers("cdp"))
        for k, (cdp, idx) in enumerate(groups):
            context.report(k / max(len(groups), 1))
            found = scan_depth_moveout(data.traces[idx], data.times, data.header("offset")[idx], targets,
                                       coefficients, self["window"])
            if not found:
                continue
            x = float(np.mean(data.header("cdpx")[idx]))
            for p in found:
                p.update(cdp=cdp, x=x)
            picks.extend(found); positions.append(x)
            updates.append(np.interp(model.z, [p["depth"] for p in found],
                                      [1 / np.sqrt(1 + p["a"]) - 1 for p in found]))
            if self["method"] == "straight-ray tomography":
                for p in found:
                    z, a = p["depth"], p["a"]
                    zero = 2 * straight_ray(model, x, x, z)
                    v = float(model.at(np.array([z]), np.array([x]))[0])
                    for offset in np.unique(np.abs(data.header("offset")[idx])):
                        h = offset / 2
                        if h == 0:
                            continue
                        cosine = z / np.hypot(z, h)
                        path = straight_ray(model, x - h, x, z) + straight_ray(model, x + h, x, z)
                        matrices.append(path - cosine * zero)
                        residuals.append(2 * (np.sqrt(max(z*z + a*h*h, 0)) - z) * cosine / v)
        if not picks:
            raise ValueError("no coherent interior RMO maximum; adjust depth window, aperture or scan range")
        if self["method"] == "vertical":
            order = np.argsort(positions)
            table = np.asarray(updates)[order]
            change = np.stack([np.interp(model.x, np.asarray(positions)[order], table[:, iz]) for iz in range(model.shape[0])])
            change = np.clip(change, -self["max_change"], self["max_change"])
            values = model.values * (1 + change)
            residual_after = None
        else:
            if not matrices:
                raise ValueError("no nonzero offsets for tomographic update")
            matrix = vstack(matrices).tocsr() @ diags(1 / model.values.ravel())
            d2, d1 = roughness_operators(np.ones(model.shape, dtype=bool))
            delta = damped_update(matrix, residuals, d2, d1, smoothing=self["smoothing"], damping=self["damping"],
                                  max_change=self["max_change"])
            values = model.values / (1 + delta.reshape(model.shape))
            residual_after = np.asarray(residuals) - matrix @ delta
        new = DepthModel(values, model.dz, model.dx, model.z0, model.x0)
        context.report(1)
        context.results[self.name] = {"picks": picks, "method": self["method"], "starting_model": model.values.copy(),
                                      "x": model.x, "z": model.z,
                                      "updated_model": values, "linear_residual_before": residuals,
                                      "linear_residual_after": residual_after,
                                      "next_step": "Re-migrate original data; linear residuals are not a re-imaged QC."}
        context.depth_model = new
        return model_section(new, data.name + " MVA velocity", data.history)
