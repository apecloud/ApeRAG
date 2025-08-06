import { Configuration, DefaultApi, GraphApi, QuotasApi } from '@/api';
import { getAuthorizationHeader } from '@/models/user';
import axios from 'axios';
import { toast } from 'react-toastify';
import { getLocale, history, getIntl } from 'umi';

export const request = axios.create({
  baseURL: '',
  timeout: 1000 * 5,
});

request.interceptors.request.use(
  (config) => {
    const lang = getLocale();
    Object.assign(config.headers, {
      lang,
      ...getAuthorizationHeader(),
    });
    config.headers['lang'] = lang;
    return config;
  },
  function (error) {
    // Do something with request error
    return Promise.reject(error);
  },
);

request.interceptors.response.use(
  function (response) {
    // Any status code that lie within the range of 2xx cause this function to trigger
    // Do something with response data

    return response;
  },
  function (err: any) {
    // Any status codes that falls outside the range of 2xx cause this function to trigger
    // Do something with response error
    if (err.status === 401) {
      // Don't redirect if we're already on the signin page to avoid infinite redirects
      const currentPath = window.location.pathname;
      if (!currentPath.includes('/accounts/signin')) {
        history.replace(
          `/accounts/signin?redirectUri=${encodeURIComponent(
            window.location.href,
          )}`,
        );
      }
    } else {
      // Handle quota exceeded errors with friendly messages
      const errorData = err.response?.data;
      let errorMessage = err.response?.data?.detail || err.message || 'request error';
      
      if (errorData && errorData.details?.quota_exceeded) {
        const intl = getIntl();
        const quotaType = errorData.details.quota_type;
        const currentUsage = errorData.details.current_usage;
        const quotaLimit = errorData.details.quota_limit;
        
        // Map quota types to user-friendly messages
        const quotaTypeMessages: Record<string, string> = {
          'max_collection_count': intl.formatMessage({ id: 'quota.error.collection_exceeded' }),
          'max_document_count': intl.formatMessage({ id: 'quota.error.document_exceeded' }),
          'max_document_count_per_collection': intl.formatMessage({ id: 'quota.error.document_exceeded' }),
          'max_bot_count': intl.formatMessage({ id: 'quota.error.bot_exceeded' }),
        };
        
        const friendlyMessage = quotaTypeMessages[quotaType] || intl.formatMessage({ id: 'quota.error.exceeded' });
        
        if (currentUsage !== null && quotaLimit) {
          const usageDetail = intl.formatMessage(
            { id: 'quota.error.current_usage' },
            { current: currentUsage, limit: quotaLimit }
          );
          errorMessage = `${friendlyMessage}。${usageDetail}。${intl.formatMessage({ id: 'quota.error.upgrade_hint' })}`;
        } else {
          errorMessage = `${friendlyMessage}。${intl.formatMessage({ id: 'quota.error.upgrade_hint' })}`;
        }
      } else if (errorData?.error_code && (
        errorData.error_code === 'COLLECTION_QUOTA_EXCEEDED' ||
        errorData.error_code === 'DOCUMENT_QUOTA_EXCEEDED' ||
        errorData.error_code === 'BOT_QUOTA_EXCEEDED'
      )) {
        // Fallback for quota errors without detailed info
        const intl = getIntl();
        const errorCodeMessages: Record<string, string> = {
          'COLLECTION_QUOTA_EXCEEDED': intl.formatMessage({ id: 'quota.error.collection_exceeded' }),
          'DOCUMENT_QUOTA_EXCEEDED': intl.formatMessage({ id: 'quota.error.document_exceeded' }),
          'BOT_QUOTA_EXCEEDED': intl.formatMessage({ id: 'quota.error.bot_exceeded' }),
        };
        
        const friendlyMessage = errorCodeMessages[errorData.error_code] || intl.formatMessage({ id: 'quota.error.exceeded' });
        errorMessage = `${friendlyMessage}。${intl.formatMessage({ id: 'quota.error.upgrade_hint' })}`;
      }
      
      toast.error(errorMessage);
    }
    return Promise.reject(err);
  },
);

const requestConfiguration = new Configuration();

export const api = new DefaultApi(requestConfiguration, '/api/v1', request);
export const graphApi = new GraphApi(requestConfiguration, '/api/v1', request);
export const quotasApi = new QuotasApi(requestConfiguration, '/api/v1', request);
