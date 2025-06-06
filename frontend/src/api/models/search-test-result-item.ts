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



/**
 * 
 * @export
 * @interface SearchTestResultItem
 */
export interface SearchTestResultItem {
    /**
     * Result rank
     * @type {number}
     * @memberof SearchTestResultItem
     */
    'rank'?: number;
    /**
     * Result score
     * @type {number}
     * @memberof SearchTestResultItem
     */
    'score'?: number;
    /**
     * Result content
     * @type {string}
     * @memberof SearchTestResultItem
     */
    'content'?: string;
    /**
     * Source document or metadata
     * @type {string}
     * @memberof SearchTestResultItem
     */
    'source'?: string;
    /**
     * Recall type
     * @type {string}
     * @memberof SearchTestResultItem
     */
    'recall_type'?: SearchTestResultItemRecallTypeEnum;
}

export const SearchTestResultItemRecallTypeEnum = {
    vector_search: 'vector_search',
    graph_search: 'graph_search',
    fulltext_search: 'fulltext_search'
} as const;

export type SearchTestResultItemRecallTypeEnum = typeof SearchTestResultItemRecallTypeEnum[keyof typeof SearchTestResultItemRecallTypeEnum];


