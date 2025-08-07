import { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'umi';
import { Spin } from 'antd';

const OAuthCallback = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    const handleOAuth = async () => {
      try {
        // Check for error parameters first
        const error = searchParams.get('error');
        if (error) {
          navigate('/accounts/signin?error=oauth_failed');
          return;
        }

        // Get OAuth parameters from URL
        const code = searchParams.get('code');
        const state = searchParams.get('state');

        if (!code || !state) {
          navigate('/accounts/signin?error=oauth_invalid');
          return;
        }

        // Determine provider from localStorage first, then fallback to referrer detection
        let provider = localStorage.getItem('oauth_provider');
        
        // If no provider in localStorage, try to determine from referrer or state
        if (!provider) {
          const referrer = document.referrer;
          
          // Try to detect provider from referrer URL
          if (referrer.includes('github.com')) {
            provider = 'github';
          } else if (referrer.includes('google.com') || referrer.includes('accounts.google.com')) {
            provider = 'google';
          } else {
            // Default fallback
            provider = 'github';
          }
        }
        
        // Clean up the stored provider
        localStorage.removeItem('oauth_provider');

        // Construct the callback URL with all parameters
        const callbackUrl = `/api/v1/auth/${provider}/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;

        try {
          // Use fetch with proper error handling for OAuth callback
          const response = await fetch(callbackUrl, {
            method: 'GET',
            credentials: 'include', // Important for cookies
            redirect: 'manual', // Handle redirects manually
          });

          // Check for successful response (2xx) - fastapi-users cookie auth returns 204 No Content
          if (response.ok) {
            // For cookie authentication, we expect 204 No Content with Set-Cookie header
            if (response.status === 204) {
              // Wait a moment to ensure cookie is properly set before redirecting
              setTimeout(() => {
                navigate('/');
              }, 1000);
              return;
            }
            
            // Handle other successful responses
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
              await response.json();
            }
            
            // Authentication successful, redirect to main application
            navigate('/');
            return;
          }

          // Check if it's a redirect response (less likely with cookie auth but handle anyway)
          if (response.status >= 300 && response.status < 400) {
            const location = response.headers.get('location');
            
            // If there's a redirect location, follow it
            if (location) {
              if (location.startsWith('/')) {
                // Relative redirect - navigate within the app
                navigate(location);
              } else {
                // Absolute redirect - use window.location
                window.location.href = location;
              }
            } else {
              // No location header, assume success and redirect to home
              navigate('/');
            }
            return;
          }

          // Handle error responses
          console.error('OAuth callback failed with status:', response.status);
          
          // Try to get error details from response
          let errorType = 'oauth_failed';
          try {
            const errorData = await response.text();
            console.error('OAuth callback error details:', errorData);
            
            // Check for specific error types based on status code
            if (response.status === 400) {
              errorType = 'oauth_invalid';
            } else if (response.status === 401 || response.status === 403) {
              errorType = 'oauth_unauthorized';
            } else if (response.status >= 500) {
              errorType = 'oauth_server_error';
            }
          } catch (parseError) {
            console.error('Failed to parse error response:', parseError);
          }
          
          navigate(`/accounts/signin?error=${errorType}`);
        } catch (fetchError) {
          console.error('OAuth callback fetch error:', fetchError);
          // Fallback: try direct navigation to the callback URL
          window.location.href = callbackUrl;
        }
      } catch (error) {
        console.error('OAuth callback general error:', error);
        navigate('/accounts/signin?error=oauth_failed');
      }
    };

    handleOAuth();
  }, [navigate, searchParams]);
  
  return (
    <div
      style={{
        height: '100vh',
        width: '100vw',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: 16,
        backgroundColor: '#f5f5f5',
      }}
    >
      <Spin size="large" />
      <div>Processing OAuth login...</div>
    </div>
  );
};

export default OAuthCallback;
