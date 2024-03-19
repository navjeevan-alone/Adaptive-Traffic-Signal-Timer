import tensorflow as tf
import pickle
import numpy as np
from .misc import show
import os

def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    S, B, C = m['side'], m['num'], m['classes']
    SS = S * S  # number of grid cells

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tside    = {}'.format(m['side']))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, SS, C]
    size2 = [None, SS, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
        'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
        'areas': _areas, 'upleft': _upleft, 'botright': _botright
    }

    # Extract the coordinate prediction from net_out
    coords = net_out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])
    wh = tf.math.pow(coords[:, :, :, 2:4], 2) * S  # unit: grid cell
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]  # unit: grid cell^2
    centers = coords[:, :, :, 0:2]  # [batch, SS, B, 2]
    floor = centers - (wh * .5)  # [batch, SS, B, 2]
    ceil = centers + (wh * .5)  # [batch, SS, B, 2]

    # calculate the intersection areas
    intersect_upleft = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = intersect_wh[:, :, :, 0] * intersect_wh[:, :, :, 1]

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.cast(best_box, tf.float32)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    proid = sprob * _proid

    # flatten 'em all
    probs = tf.reshape(_probs, [-1])
    proid = tf.reshape(proid, [-1])
    confs = tf.reshape(confs, [-1])
    conid = tf.reshape(conid, [-1])
    coord = tf.reshape(_coord, [-1])
    cooid = tf.reshape(cooid, [-1])

    self.fetch += [probs, confs, conid, cooid, proid]
    true = tf.concat([probs, confs, coord], 1)
    wght = tf.concat([proid, conid, cooid], 1)
    print('Building {} loss'.format(m['model']))
    loss = tf.math.square(net_out - true)
    loss = tf.multiply(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)
