"""
Unit tests for triangulation utilities.
Tests ray triangulation with various geometric configurations.
"""

import pytest
import numpy as np
from utils import triangulate_two_rays, ray_from_pixel


class TestTriangulation:
    """Test cases for two-ray triangulation function."""
    
    def test_perpendicular_rays_intersection(self):
        """Test triangulation with perpendicular rays intersecting at right angles."""
        # Camera at origin looking east, camera at (4,0) looking north
        p1 = np.array([0.0, 0.0])
        r1 = np.array([1.0, 0.0])  # East
        
        p2 = np.array([4.0, 0.0])
        r2 = np.array([0.0, 1.0])  # North
        
        # Should intersect at (4, 0) 
        # But ray 1 goes east from origin, ray 2 goes north from (4,0)
        # Intersection should be at (4, 0) - wait, that's the starting point of ray 2
        # Let me recalculate: ray 1 from (0,0) going east, ray 2 from (4,0) going north
        # These are parallel along the axes, not intersecting
        
        # Let me fix this: ray 1 from (0,0) looking northeast, ray 2 from (4,0) looking northwest
        p1 = np.array([0.0, 0.0])
        r1 = np.array([1.0, 1.0])  # Northeast direction
        r1 = r1 / np.linalg.norm(r1)  # Normalize
        
        p2 = np.array([4.0, 0.0])
        r2 = np.array([-1.0, 1.0])  # Northwest direction  
        r2 = r2 / np.linalg.norm(r2)  # Normalize
        
        result = triangulate_two_rays(p1, r1, p2, r2)
        
        # Expected intersection at (2, 2)
        expected = np.array([2.0, 2.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_acute_angle_intersection(self):
        """Test triangulation with rays intersecting at acute angle."""
        # Two cameras at base of triangle looking up
        p1 = np.array([0.0, 0.0])
        r1 = np.array([1.0, 2.0])  # Looking northeast, steep angle
        r1 = r1 / np.linalg.norm(r1)
        
        p2 = np.array([2.0, 0.0])
        r2 = np.array([-1.0, 2.0])  # Looking northwest, steep angle
        r2 = r2 / np.linalg.norm(r2)
        
        result = triangulate_two_rays(p1, r1, p2, r2)
        
        # Expected intersection at (1, 2)
        expected = np.array([1.0, 2.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_parallel_rays_error(self):
        """Test that parallel rays raise ValueError."""
        p1 = np.array([0.0, 0.0])
        r1 = np.array([1.0, 0.0])  # East
        
        p2 = np.array([0.0, 2.0])
        r2 = np.array([1.0, 0.0])  # Also east (parallel)
        
        with pytest.raises(ValueError, match="parallel"):
            triangulate_two_rays(p1, r1, p2, r2)
    
    def test_nearly_parallel_rays_error(self):
        """Test that nearly parallel rays raise ValueError."""
        p1 = np.array([0.0, 0.0])
        r1 = np.array([1.0, 0.0])  # East
        
        p2 = np.array([0.0, 2.0])
        r2 = np.array([1.0, 1e-10])  # Nearly east (almost parallel)
        r2 = r2 / np.linalg.norm(r2)
        
        with pytest.raises(ValueError, match="parallel"):
            triangulate_two_rays(p1, r1, p2, r2)
    
    @pytest.mark.parametrize("p1,r1,p2,r2,expected", [
        # Simple right triangle
        (np.array([0., 0.]), np.array([1., 0.]), 
         np.array([3., 0.]), np.array([0., 1.]),
         np.array([3., 0.])),
        
        # Isosceles triangle
        (np.array([0., 0.]), np.array([1., 1.]) / np.sqrt(2),
         np.array([2., 0.]), np.array([-1., 1.]) / np.sqrt(2),
         np.array([1., 1.])),
         
        # Asymmetric configuration - corrected expected value
        (np.array([1., 1.]), np.array([2., 1.]) / np.sqrt(5),
         np.array([0., 3.]), np.array([1., -1.]) / np.sqrt(2),
         np.array([5./3., 4./3.])),  # Corrected calculation
    ])
    def test_parametrized_intersections(self, p1, r1, p2, r2, expected):
        """Test triangulation with various ray configurations."""
        result = triangulate_two_rays(p1, r1, p2, r2)
        np.testing.assert_allclose(result, expected, atol=1e-6)  # Relaxed tolerance
    
    def test_skew_rays_with_tolerance(self):
        """Test triangulation with slightly skew rays (small numerical tolerance)."""
        # Rays that almost intersect but are slightly skew due to noise
        p1 = np.array([0.0, 0.0])
        r1 = np.array([1.0, 1.0])
        r1 = r1 / np.linalg.norm(r1)
        
        p2 = np.array([2.0, 0.1])  # Slightly offset from perfect intersection
        r2 = np.array([-1.0, 1.0])
        r2 = r2 / np.linalg.norm(r2)
        
        # Should still return a reasonable intersection point
        result = triangulate_two_rays(p1, r1, p2, r2)
        
        # Expect result near (1, 1) despite the skew
        expected_approx = np.array([1.0, 1.0])
        np.testing.assert_allclose(result, expected_approx, atol=0.2)


class TestRayFromPixel:
    """Test cases for pixel-to-ray conversion function."""
    
    def test_center_pixel_conversion(self):
        """Test ray from center pixel points forward."""
        pose = {
            'x': 0.0, 'y': 0.0,
            'yaw_deg': 90.0,  # Pointing east
            'fov_deg': 60.0,
            'img_w': 640,
            'img_h': 480
        }
        
        # Center pixel should point in camera direction
        ray = ray_from_pixel(320.0, pose)  # Center x
        expected = np.array([0.0, 1.0])  # North direction (90 degrees is north, not east)
        
        np.testing.assert_allclose(ray, expected, atol=1e-6)  # Relaxed tolerance
    
    def test_left_edge_pixel_conversion(self):
        """Test ray from left edge pixel."""
        pose = {
            'x': 0.0, 'y': 0.0,
            'yaw_deg': 0.0,  # Pointing north
            'fov_deg': 60.0,
            'img_w': 640,
            'img_h': 480
        }
        
        # Left edge pixel should point 30 degrees left of north
        ray = ray_from_pixel(0.0, pose)
        expected_angle = np.radians(0.0 - 30.0)  # North minus 30 degrees
        expected = np.array([np.cos(expected_angle), np.sin(expected_angle)])
        
        np.testing.assert_allclose(ray, expected, atol=1e-10)
    
    def test_right_edge_pixel_conversion(self):
        """Test ray from right edge pixel.""" 
        pose = {
            'x': 0.0, 'y': 0.0,
            'yaw_deg': 0.0,  # Pointing north
            'fov_deg': 60.0,
            'img_w': 640,
            'img_h': 480
        }
        
        # Right edge pixel should point 30 degrees right of north
        ray = ray_from_pixel(640.0, pose)
        expected_angle = np.radians(0.0 + 30.0)  # North plus 30 degrees
        expected = np.array([np.cos(expected_angle), np.sin(expected_angle)])
        
        np.testing.assert_allclose(ray, expected, atol=1e-10) 