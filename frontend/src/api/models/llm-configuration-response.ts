/* tslint:disable */
/* eslint-disable */
/**
 * ApeRAG API
 * ApeRAG API Documentation
 *
 * The version of the OpenAPI document: 1.0.0
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


// May contain unused imports in some cases
// @ts-ignore
import type { LlmProvider } from './llm-provider';
// May contain unused imports in some cases
// @ts-ignore
import type { LlmProviderModel } from './llm-provider-model';

/**
 * 
 * @export
 * @interface LlmConfigurationResponse
 */
export interface LlmConfigurationResponse {
    /**
     * List of LLM providers
     * @type {Array<LlmProvider>}
     * @memberof LlmConfigurationResponse
     */
    'providers': Array<LlmProvider>;
    /**
     * List of LLM provider models
     * @type {Array<LlmProviderModel>}
     * @memberof LlmConfigurationResponse
     */
    'models': Array<LlmProviderModel>;
}

