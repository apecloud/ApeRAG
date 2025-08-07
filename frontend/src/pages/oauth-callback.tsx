import { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'umi';
import { Spin, Alert } from 'antd';

// Add immediate console log to check if file is loaded
console.log('OAuth callback file loaded');

const OAuthCallback = () => {
  console.log('OAuth callback component instantiated');
  
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Add immediate effect to check component mounting
  useEffect(() => {
    console.log('OAuth callback useEffect triggered');
    console.log('OAuth callback page loaded');
    console.log('Current URL:', window.location.href);
    console.log('Search params:', window.location.search);
    console.log('searchParams object:', searchParams);
    
    const handleOAuth = async () => {
      console.log('handleOAuth function started');
      try {
        // Check for error parameters first
        const error = searchParams.get('error');
        if (error) {
          console.error('OAuth error:', error);
          const errorDescription = searchParams.get('error_description');
          if (errorDescription) {
            console.error('OAuth error description:', errorDescription);
          }
          navigate('/accounts/signin?error=oauth_failed');
          return;
        }

        // Get OAuth parameters from URL
        const code = searchParams.get('code');
        const state = searchParams.get('state');

        if (!code || !state) {
          console.error('Missing OAuth parameters');
          navigate('/accounts/signin?error=oauth_invalid');
          return;
        }

        // Determine provider from localStorage first, then fallback to referrer detection
        let provider = localStorage.getItem('oauth_provider');
        
        console.log('OAuth callback - stored provider:', provider);
        console.log('OAuth callback - code:', code);
        console.log('OAuth callback - state:', state);
        
        // If no provider in localStorage, try to determine from referrer or state
        if (!provider) {
          const referrer = document.referrer;
          console.log('OAuth callback - referrer:', referrer);
          
          // Try to detect provider from referrer URL
          if (referrer.includes('github.com')) {
            provider = 'github';
          } else if (referrer.includes('google.com') || referrer.includes('accounts.google.com')) {
            provider = 'google';
          } else {
            // Default fallback - this might need adjustment based on your setup
            provider = 'github';
            console.warn('Could not determine OAuth provider, defaulting to github');
          }
        }
        
        console.log('OAuth callback - using provider:', provider);
        
        // Clean up the stored provider
        localStorage.removeItem('oauth_provider');

        // Construct the callback URL with all parameters
        const callbackUrl = `/api/v1/auth/${provider}/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;
        
        console.log('OAuth callback - calling:', callbackUrl);

        try {
          // Use fetch with proper error handling for OAuth callback
          const response = await fetch(callbackUrl, {
            method: 'GET',
            credentials: 'include', // Important for cookies
            redirect: 'manual', // Handle redirects manually
          });

          console.log('OAuth callback response status:', response.status);
          console.log('OAuth callback response headers:', Object.fromEntries(response.headers.entries()));

          // Check for successful response (2xx) - fastapi-users cookie auth returns 204 No Content
          if (response.ok) {
            console.log('OAuth callback successful - status:', response.status);
            
            // For cookie authentication, we expect 204 No Content with Set-Cookie header
            if (response.status === 204) {
              console.log('OAuth authentication successful - cookie should be set');
              console.log('Set-Cookie headers:', response.headers.get('set-cookie'));
              
              // Wait a moment to ensure cookie is properly set before redirecting
              setTimeout(() => {
                console.log('Redirecting to main application after cookie delay');
                navigate('/');
              }, 1000);
              return;
            }
            
            // Handle other successful responses
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
              const data = await response.json();
              console.log('OAuth callback response data:', data);
            }
            
            // Authentication successful, redirect to main application
            navigate('/');
            return;
          }

          // Check if it's a redirect response (less likely with cookie auth but handle anyway)
          if (response.status >= 300 && response.status < 400) {
            const location = response.headers.get('location');
            console.log('OAuth callback redirect location:', location);
            
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
          const errorText = await response.text();
          console.error('OAuth callback error:', response.status, errorText);
          navigate('/accounts/signin?error=oauth_failed');
        } catch (fetchError) {
          console.error('OAuth callback fetch error:', fetchError);
          // Fallback: try direct navigation to the callback URL
          console.log('Falling back to direct navigation');
          window.location.href = callbackUrl;
        }
      } catch (error) {
        console.error('Error handling OAuth callback:', error);
        navigate('/accounts/signin?error=oauth_failed');
      }
    };

    handleOAuth();
  }, [navigate, searchParams]);

  console.log('OAuth callback component rendering');
  
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
      <div style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
        Debug: Check console for logs
      </div>
    </div>
  );
};

export default OAuthCallback;
