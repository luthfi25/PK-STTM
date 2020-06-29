package models;

public class GT_POINTS {
    private double x;
    private double y;

    public GT_POINTS(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    @Override
    public String toString() {
        return "GT_POINTS{" +
                "x=" + x +
                ", y=" + y +
                '}';
    }

    //    public double map(int id, double x) {
//        int low = src_points[id].size()-1;
//        int max = src_points[id].size()-1;
//        int high = 0;
//
//        while (high <= low) {
//            int mid = (low + high) / 2;
//            if (src_points[id][mid].x == x) {
//                return src_points[id][mid].y;
//            }
//            if (src_points[id][low].x == x) {
//                return src_points[id][low].y;
//            }
//            if (src_points[id][high].x == x) {
//                return src_points[id][high].y;
//            }
//            if (low == high + 1) {
//                mid = high;
//            }
//            if (src_points[id][mid].x > x && x > src_points[id][Min(mid+1, max)].x) {
//                double x_s = src_points[id][Min(mid+1, max)].x;
//                double x_l = src_points[id][mid].x;
//                double y_s = src_points[id][Min(mid+1, max)].y;
//                double y_l = src_points[id][mid].y;
//                double x_gap = x_l - x_s;
//                double y_gap = y_l - y_s;
//                double frac = (x - x_s) / x_gap;
//                return y_s + frac*y_gap;
//            }
//            else if (src_points[id][mid].x > x) {
//                high = mid;
//            }
//            else {
//                low = mid;
//            }
//        }
//        return 0.0;
//    }

}
