
class LINE_SEGMENT
{
private:
	std::vector<cv::Point> _pts;
	float _degree;				 //[0,180]
	cv::Point2f _centroid;
public:
	LINE_SEGMENT()
	{
		reset();
	}
	~LINE_SEGMENT()
	{

	}
public:
	float degree() {
		if (_degree < 0)
			fit_degree();
		return _degree;
	}
	cv::Point2f centroid()
	{
		if (_centroid.x < 0 || _centroid.y < 0)
			fit_centroid();
		return _centroid;
	}
	int length()
	{
		return _pts.size();
	}
	void reset(bool keep_pts = false)
	{
		if(!keep_pts)
			_pts.clear();
		_degree = -100;
		_centroid.x = _centroid.y = -1;;
	}
	void add_pts(cv::Point pt)
	{
		reset(true);
		_pts.push_back(pt);
	}
private:
	void fit_centroid()
	{

		if (_pts.empty()) return;

		_centroid.x = 0;
		_centroid.y = 0;
		for (int k = 0; k < _pts.size(); k++)
		{
			_centroid.x += _pts[k].x;
			_centroid.y += _pts[k].y;
		}

		_centroid.x /= _pts.size();
		_centroid.y /= _pts.size();
		return;

	}
	void fit_degree()
	{
		if (_pts.empty())
		{
			_degree = -100;
			return;
		}
		cv::Point lt = _pts[0], br = _pts[0];

		for (int k = 1; k < _pts.size(); k++)
		{
			if (lt.x > _pts[k].x) lt = _pts[k];
			if (br.x < _pts[k].x) br = _pts[k];
		}
		if (lt.x == br.x && lt.y == br.y) _degree = 90;
		else if (lt.x == br.x) _degree = 90;
		else
		{
			float cx = centroid().x;
			float cy = centroid().y;
			float dx = (br.x - cx) - (lt.x - cx);
			float dy = (cy - br.y) - (cy - lt.y);
			float angle = atan2(dy, dx) * 180 / M_PI;
			if (angle < 0) angle += 180;
			_degree = angle;
		}
	}
public:
	cv::Point& operator[](int pos)
	{
		return _pts[pos];
	}
};

class CURVE_FITTING
{
	int _order;
	std::vector<float> _coefs;
public:
	CURVE_FITTING(int order)
	{
		_order = order;
	}
	~CURVE_FITTING()
	{

	}
private:
	float solver_for_y(float x) 
	{
		float x_order = 1.0;
		float sum = 0.0f;
		for (int k = 0; k < _coefs.size(); k++)
		{
			sum += _coefs[k] * x_order;
			x_order *= x;
		}
		return sum;
	}

	bool LSM(std::vector<cv::Point> &pts, int order, std::vector<float>& coefs)

	{
		coefs.clear();

		std::vector<float> py, px;
		py.resize(pts.size());
		px.resize(pts.size() * order);

		for (auto itr = pts.begin(); itr != pts.end(); ++itr)
		{
			int i = std::distance(pts.begin(),itr);

			py[i] = (*itr).y;

			int j = 0;

			while (j < order) {

				px[order*i + j] = pow(((*itr).x), float(j));

				j++;

			}
		}

		cv::Mat X = cv::Mat(pts.size(), order, CV_32FC1);
		for (int row = 0; row < pts.size(); row++)
		{
			for (int col = 0; col < order; col++)
			{
				X.at<float>(row, col) = px[row * order + col];
			}
		}



		cv::Mat X_T;

		transpose(X, X_T);

		cv::Mat Y = cv::Mat(py,true);

		cv::Mat para = ((X_T*X).inv())*X_T*Y;

		for (int s = 0; s < order; s++) {

			coefs.push_back(para.at<float>(s));

		}

		return true;

	}
public:
	LINE_SEGMENT* run(std::vector< cv::Point>& pts)
	{

		if (pts.empty() < 0) return 0;
		_coefs.clear();
		LSM(pts, _order, _coefs);
		cv::RotatedRect rrect = cv::minAreaRect(pts);
		cv::Rect bbox = rrect.boundingRect();
		LINE_SEGMENT* line = new LINE_SEGMENT();
		for (int x = bbox.tl().x; x < bbox.br().x; x++)
		{
			int y = solver_for_y(x);
			line->add_pts(cv::Point(x, y));
		}		
		return line;
	}

};
